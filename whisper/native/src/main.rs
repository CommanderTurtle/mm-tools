use clap::{Parser, ValueEnum};
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

#[derive(Clone, Debug, ValueEnum)]
enum Operation {
    Both,
    Verbatim,
    Intended,
    Verbatimize,
    Align,
}

impl Operation {
    fn value(&self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::Verbatim => "verbatim",
            Self::Intended => "intended",
            Self::Verbatimize => "verbatimize",
            Self::Align => "align",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Transcribe media with the isolated local CrisperWhisper runtime")]
struct Args {
    input: PathBuf,
    #[arg(long, value_enum, default_value = "both")]
    operation: Operation,
    #[arg(long, default_value = "en")]
    language: String,
    #[arg(long, default_value = "")]
    transcript: String,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    no_timestamps: bool,
    #[arg(long = "hotword")]
    hotwords: Vec<String>,
}

fn is_root(path: &Path) -> bool {
    path.join("pyproject.toml").is_file() && path.join("local_app/cli.py").is_file()
}

fn find_root() -> Result<PathBuf, String> {
    if let Ok(value) = env::var("CW2_ROOT") {
        let path = PathBuf::from(value);
        if is_root(&path) {
            return Ok(path);
        }
        return Err("CW2_ROOT does not point to a CrisperWhisper Local checkout".into());
    }
    for start in [env::current_dir().ok(), env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf))].into_iter().flatten() {
        for candidate in start.ancestors() {
            if is_root(candidate) {
                return Ok(candidate.to_path_buf());
            }
        }
    }
    Err("Could not locate the checkout. Run cw2 inside it or set CW2_ROOT.".into())
}

fn main() -> ExitCode {
    let args = Args::parse();
    let root = match find_root() {
        Ok(root) => root,
        Err(message) => { eprintln!("cw2: {message}"); return ExitCode::FAILURE; }
    };
    if !root.join(".venv/bin/python").is_file() {
        eprintln!("cw2: missing isolated environment; run {}/uvsetup.sh", root.display());
        return ExitCode::FAILURE;
    }

    let mut command = Command::new("uv");
    command.current_dir(&root).args([
        "run", "--no-sync", "python", "-m", "local_app.cli",
    ]).arg(&args.input).args(["--operation", args.operation.value(), "--language", &args.language]);
    if !args.transcript.is_empty() { command.args(["--transcript", &args.transcript]); }
    if args.no_timestamps { command.arg("--no-timestamps"); }
    if let Some(output) = &args.output { command.arg("--output").arg(output); }
    for hotword in &args.hotwords { command.arg("--hotword").arg(hotword); }
    match command.status() {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1) as u8),
        Err(error) => { eprintln!("cw2: could not start uv: {error}"); ExitCode::FAILURE }
    }
}
