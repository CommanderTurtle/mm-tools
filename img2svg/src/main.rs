use std::{
    collections::VecDeque,
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use image::{imageops::FilterType, DynamicImage, ImageFormat, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use tempfile::tempdir;
use tokio::net::TcpListener;
use tower_http::{limit::RequestBodyLimitLayer, trace::TraceLayer};
use visioncortex::PathSimplifyMode;
use vtracer::{ColorMode, Config, Hierarchical};

const INDEX_HTML: &str = include_str!("../web/index.html");
const APP_CSS: &str = include_str!("../web/app.css");
const APP_JS: &str = include_str!("../web/app.js");
const MAX_UPLOAD_BYTES: usize = 64 * 1024 * 1024;

#[derive(Parser)]
#[command(name = "img2svg", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert one image or an entire directory tree.
    Convert(ConvertArgs),
    /// Start the private local vectorization studio.
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 417)]
        port: u16,
    },
}

#[derive(Args)]
struct ConvertArgs {
    /// Raster image or directory containing raster images.
    input: PathBuf,
    /// SVG file for one input, or output directory for batch input.
    output: PathBuf,
    #[command(flatten)]
    options: VectorOptions,
    /// Walk nested directories when INPUT is a directory.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    recursive: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum TraceMode {
    Color,
    Mono,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum TracePreset {
    Logo,
    Poster,
    Photo,
    Pixel,
}

#[derive(Clone, Debug, Deserialize, Serialize, Args)]
#[serde(default)]
struct VectorOptions {
    #[arg(long, value_enum, default_value = "color")]
    mode: TraceMode,
    #[arg(long, value_enum, default_value = "logo")]
    preset: TracePreset,
    /// Exact palette size for color mode (2-32).
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u8).range(2..=32))]
    colors: u8,
    /// Curve detail: 1 keeps detail, 5 produces smoother/smaller paths.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=5))]
    smoothness: u8,
    /// Discard isolated regions smaller than this many pixels.
    #[arg(long, default_value_t = 4)]
    speckle: usize,
    /// Monochrome luminance cutoff (0-255).
    #[arg(long, default_value_t = 128)]
    threshold: u8,
    /// Invert monochrome foreground/background.
    #[arg(long, default_value_t = false)]
    invert: bool,
    /// Make pixels near the corner-derived background color transparent.
    #[arg(long, default_value_t = false)]
    remove_background: bool,
    /// RGB distance tolerance used by background removal.
    #[arg(long, default_value_t = 24)]
    background_tolerance: u8,
    /// Downscale before tracing; zero preserves source dimensions.
    #[arg(long, default_value_t = 4096)]
    max_dimension: u32,
    /// SVG numeric coordinate precision.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(0..=6))]
    path_precision: u8,
}

impl Default for VectorOptions {
    fn default() -> Self {
        Self {
            mode: TraceMode::Color,
            preset: TracePreset::Logo,
            colors: 8,
            smoothness: 2,
            speckle: 4,
            threshold: 128,
            invert: false,
            remove_background: false,
            background_tolerance: 24,
            max_dimension: 4096,
            path_precision: 2,
        }
    }
}

#[derive(Debug, Serialize)]
struct VectorResult {
    svg: String,
    source_width: u32,
    source_height: u32,
    traced_width: u32,
    traced_height: u32,
    input_bytes: usize,
    output_bytes: usize,
    path_count: usize,
    elapsed_ms: u128,
}

#[derive(Clone, Default)]
struct AppState;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "img2svg=info,tower_http=info".into()),
        )
        .init();

    match Cli::parse().command {
        Command::Convert(args) => run_convert(args),
        Command::Serve { host, port } => serve(&host, port).await,
    }
}

fn run_convert(args: ConvertArgs) -> Result<()> {
    if args.input.is_file() {
        let data = fs::read(&args.input)
            .with_context(|| format!("cannot read {}", args.input.display()))?;
        let result = vectorize(&data, &args.options)?;
        if let Some(parent) = args.output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&args.output, result.svg)?;
        eprintln!(
            "{} -> {} ({} paths, {} bytes, {} ms)",
            args.input.display(),
            args.output.display(),
            result.path_count,
            result.output_bytes,
            result.elapsed_ms
        );
        return Ok(());
    }
    if !args.input.is_dir() {
        bail!("input does not exist: {}", args.input.display());
    }
    fs::create_dir_all(&args.output)?;
    let mut queue = VecDeque::from([args.input.clone()]);
    let mut converted = 0usize;
    while let Some(dir) = queue.pop_front() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if args.recursive {
                    queue.push_back(path);
                }
                continue;
            }
            if !is_raster(&path) {
                continue;
            }
            let relative = path.strip_prefix(&args.input)?;
            let mut output = args.output.join(relative);
            output.set_extension("svg");
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            let data = fs::read(&path)?;
            match vectorize(&data, &args.options) {
                Ok(result) => {
                    fs::write(&output, result.svg)?;
                    converted += 1;
                    eprintln!("{} -> {}", path.display(), output.display());
                }
                Err(error) => eprintln!("skip {}: {error:#}", path.display()),
            }
        }
    }
    eprintln!("converted {converted} image(s)");
    Ok(())
}

fn is_raster(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp"))
        .unwrap_or(false)
}

async fn serve(host: &str, port: u16) -> Result<()> {
    let address: SocketAddr = format!("{host}:{port}").parse()?;
    let app = Router::new()
        .route("/", get(index))
        .route("/app.css", get(css))
        .route("/app.js", get(js))
        .route("/api/health", get(health))
        .route("/api/vectorize", post(vectorize_endpoint))
        .with_state(AppState)
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(MAX_UPLOAD_BYTES))
        .layer(TraceLayer::new_for_http());
    let listener = TcpListener::bind(address).await?;
    println!("img2svg studio: http://{address}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> impl IntoResponse {
    typed_content("text/html; charset=utf-8", INDEX_HTML)
}

async fn css() -> impl IntoResponse {
    typed_content("text/css; charset=utf-8", APP_CSS)
}

async fn js() -> impl IntoResponse {
    typed_content("text/javascript; charset=utf-8", APP_JS)
}

fn typed_content(content_type: &'static str, content: &'static str) -> Response {
    let mut response = Response::new(Body::from(content));
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store"),
    );
    response
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "engine": "visioncortex/vtracer",
        "network": "local-only"
    }))
}

async fn vectorize_endpoint(
    State(_): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<VectorResult>, ApiError> {
    let mut file = None;
    let mut options = VectorOptions::default();
    while let Some(field) = multipart.next_field().await.map_err(ApiError::bad_request)? {
        match field.name() {
            Some("image") => {
                let bytes = field.bytes().await.map_err(ApiError::bad_request)?;
                file = Some(bytes.to_vec());
            }
            Some("options") => {
                let text = field.text().await.map_err(ApiError::bad_request)?;
                options = serde_json::from_str(&text).map_err(ApiError::bad_request)?;
            }
            _ => {}
        }
    }
    let file = file.ok_or_else(|| ApiError::bad_request("missing image field"))?;
    if file.len() > MAX_UPLOAD_BYTES {
        return Err(ApiError(StatusCode::PAYLOAD_TOO_LARGE, "image exceeds 64 MiB".into()));
    }
    validate_options(&options).map_err(ApiError::bad_request)?;
    let result = tokio::task::spawn_blocking(move || vectorize(&file, &options))
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::bad_request)?;
    Ok(Json(result))
}

fn validate_options(options: &VectorOptions) -> Result<()> {
    if !(2..=32).contains(&options.colors) {
        bail!("colors must be between 2 and 32");
    }
    if !(1..=5).contains(&options.smoothness) {
        bail!("smoothness must be between 1 and 5");
    }
    if options.max_dimension > 16384 {
        bail!("max_dimension may not exceed 16384");
    }
    Ok(())
}

fn vectorize(bytes: &[u8], options: &VectorOptions) -> Result<VectorResult> {
    validate_options(options)?;
    let start = Instant::now();
    let original = image::load_from_memory(bytes).context("unsupported or corrupt raster image")?;
    let (source_width, source_height) = (original.width(), original.height());
    if source_width == 0 || source_height == 0 {
        bail!("image has zero width or height");
    }
    if source_width.saturating_mul(source_height) > 180_000_000 {
        bail!("image exceeds the 180 megapixel safety limit");
    }
    let mut image = resize_if_needed(original, options.max_dimension).to_rgba8();
    if options.remove_background {
        remove_corner_background(&mut image, options.background_tolerance);
    }
    match options.mode {
        TraceMode::Mono => threshold_image(&mut image, options.threshold, options.invert),
        TraceMode::Color => quantize_kmeans(&mut image, options.colors as usize),
    }
    let (traced_width, traced_height) = image.dimensions();

    let temp = tempdir()?;
    let input = temp.path().join("source.png");
    let output = temp.path().join("output.svg");
    DynamicImage::ImageRgba8(image)
        .save_with_format(&input, ImageFormat::Png)
        .context("failed to prepare trace image")?;
    let config = trace_config(options);
    vtracer::convert_image_to_svg(&input, &output, config)
        .map_err(anyhow::Error::msg)?;
    let svg = fs::read_to_string(&output).context("tracer did not produce SVG")?;
    let output_bytes = svg.len();
    let path_count = svg.match_indices("<path").count();
    Ok(VectorResult {
        svg,
        source_width,
        source_height,
        traced_width,
        traced_height,
        input_bytes: bytes.len(),
        output_bytes,
        path_count,
        elapsed_ms: start.elapsed().as_millis(),
    })
}

fn resize_if_needed(image: DynamicImage, max_dimension: u32) -> DynamicImage {
    if max_dimension == 0 || image.width().max(image.height()) <= max_dimension {
        return image;
    }
    image.resize(max_dimension, max_dimension, FilterType::Lanczos3)
}

fn trace_config(options: &VectorOptions) -> Config {
    let detail = options.smoothness as f64;
    let (corner, splice, length) = match options.preset {
        TracePreset::Logo => (55, 40, 2.0 + detail * 1.4),
        TracePreset::Poster => (70, 48, 2.5 + detail * 1.7),
        TracePreset::Photo => (180, 55, 3.0 + detail * 2.2),
        TracePreset::Pixel => (0, 0, 1.0),
    };
    Config {
        color_mode: match options.mode {
            TraceMode::Color => ColorMode::Color,
            TraceMode::Mono => ColorMode::Binary,
        },
        hierarchical: Hierarchical::Stacked,
        filter_speckle: options.speckle,
        color_precision: 8,
        layer_difference: match options.preset {
            TracePreset::Photo => 48,
            TracePreset::Poster => 24,
            _ => 12,
        },
        mode: match options.preset {
            TracePreset::Pixel => PathSimplifyMode::None,
            _ => PathSimplifyMode::Spline,
        },
        corner_threshold: corner,
        length_threshold: length,
        max_iterations: 10 + options.smoothness as usize * 2,
        splice_threshold: splice,
        path_precision: Some(options.path_precision as u32),
    }
}

fn threshold_image(image: &mut RgbaImage, threshold: u8, invert: bool) {
    for pixel in image.pixels_mut() {
        if pixel[3] == 0 {
            continue;
        }
        let luminance = (u16::from(pixel[0]) * 54
            + u16::from(pixel[1]) * 183
            + u16::from(pixel[2]) * 19)
            / 256;
        let foreground = (luminance < u16::from(threshold)) ^ invert;
        let value = if foreground { 0 } else { 255 };
        *pixel = Rgba([value, value, value, pixel[3]]);
    }
}

fn remove_corner_background(image: &mut RgbaImage, tolerance: u8) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return;
    }
    let corners = [
        image.get_pixel(0, 0),
        image.get_pixel(width - 1, 0),
        image.get_pixel(0, height - 1),
        image.get_pixel(width - 1, height - 1),
    ];
    let background = [
        (corners.iter().map(|p| u16::from(p[0])).sum::<u16>() / 4) as u8,
        (corners.iter().map(|p| u16::from(p[1])).sum::<u16>() / 4) as u8,
        (corners.iter().map(|p| u16::from(p[2])).sum::<u16>() / 4) as u8,
    ];
    let tolerance_sq = i32::from(tolerance).pow(2) * 3;
    for pixel in image.pixels_mut() {
        let distance = (i32::from(pixel[0]) - i32::from(background[0])).pow(2)
            + (i32::from(pixel[1]) - i32::from(background[1])).pow(2)
            + (i32::from(pixel[2]) - i32::from(background[2])).pow(2);
        if distance <= tolerance_sq {
            pixel[3] = 0;
        }
    }
}

/// Deterministic RGB k-means. Sampling keeps setup bounded; every opaque pixel
/// is mapped to the final palette. Alpha is preserved and transparent pixels
/// are excluded from training.
fn quantize_kmeans(image: &mut RgbaImage, colors: usize) {
    let opaque: Vec<[f32; 3]> = image
        .pixels()
        .filter(|p| p[3] > 8)
        .step_by((image.as_raw().len() / 4 / 50_000).max(1))
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    if opaque.is_empty() {
        return;
    }
    let count = colors.min(opaque.len());
    let mut centers = Vec::with_capacity(count);
    for index in 0..count {
        centers.push(opaque[index * opaque.len() / count]);
    }
    for _ in 0..12 {
        let mut sums = vec![[0f64; 3]; count];
        let mut sizes = vec![0u32; count];
        for color in &opaque {
            let nearest = nearest_center(*color, &centers);
            for channel in 0..3 {
                sums[nearest][channel] += f64::from(color[channel]);
            }
            sizes[nearest] += 1;
        }
        for index in 0..count {
            if sizes[index] == 0 {
                continue;
            }
            for channel in 0..3 {
                centers[index][channel] = (sums[index][channel] / f64::from(sizes[index])) as f32;
            }
        }
    }
    for pixel in image.pixels_mut() {
        if pixel[3] == 0 {
            continue;
        }
        let index = nearest_center(
            [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32],
            &centers,
        );
        pixel[0] = centers[index][0].round() as u8;
        pixel[1] = centers[index][1].round() as u8;
        pixel[2] = centers[index][2].round() as u8;
    }
}

fn nearest_center(color: [f32; 3], centers: &[[f32; 3]]) -> usize {
    centers
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            distance(color, **a)
                .partial_cmp(&distance(color, **b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

struct ApiError(StatusCode, String);

impl ApiError {
    fn bad_request(error: impl std::fmt::Display) -> Self {
        Self(StatusCode::BAD_REQUEST, error.to_string())
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.0,
            Json(serde_json::json!({"error": self.1})),
        )
            .into_response()
    }
}
