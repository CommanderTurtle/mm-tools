const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let currentAudio = null;
let audioContext = null;
let analyser = null;
let animationFrame = 0;
let lyricRows = [];
let activeLyricIndex = -1;
let guideModels = [];

const THEORY_OPTIONS = {
  theoryGenre: [
    ["", "Leave genre open"],
    ["alternative R&B with neo-soul harmony", "Alternative R&B · neo-soul"],
    ["art-pop with an experimental electronic edge", "Art-pop · experimental electronic"],
    ["dream-pop with shoegaze atmosphere", "Dream-pop · shoegaze"],
    ["indie folk with contemporary acoustic production", "Indie folk · contemporary acoustic"],
    ["alternative rock with post-rock dynamics", "Alternative rock · post-rock"],
    ["progressive metal with cinematic weight", "Progressive metal · cinematic heavy"],
    ["hip-hop with sample-driven soul color", "Hip-hop · sample-driven soul"],
    ["trap with sparse atmospheric production", "Trap · atmospheric"],
    ["UK garage with modern alt-pop songwriting", "UK garage · alt-pop"],
    ["drum and bass with liquid melodic writing", "Drum & bass · liquid"],
    ["house with disco and funk vocabulary", "House · disco-funk"],
    ["ambient techno with a patient cinematic arc", "Ambient techno · cinematic"],
    ["jazz fusion with contemporary electronic texture", "Jazz fusion · electronic"],
    ["cinematic orchestral music", "Cinematic orchestral"],
    ["modern chamber music with minimalist repetition", "Modern chamber · minimalism"],
    ["Americana with blues-rock weight", "Americana · blues rock"],
    ["Latin pop with live rhythmic instrumentation", "Latin pop · live rhythm"],
    ["East Asian modern pop with heritage instrumentation", "East Asian modern · heritage color"],
  ],
  theoryFusion: [
    ["", "No secondary style"],
    ["restrained dream-pop texture", "Dream-pop atmosphere"],
    ["gospel-informed harmony and backing vocals", "Gospel harmony"],
    ["jazz-informed voicings without losing the primary groove", "Jazz harmony"],
    ["chamber-orchestral detail", "Chamber orchestration"],
    ["cinematic scale and transition design", "Cinematic scale"],
    ["minimalist repetition and gradual process", "Minimalist process"],
    ["industrial percussion and abrasive texture", "Industrial texture"],
    ["ambient negative space", "Ambient space"],
    ["acoustic folk intimacy", "Acoustic intimacy"],
    ["dub-informed depth and delay", "Dub space"],
    ["psychedelic timbral movement", "Psychedelic color"],
    ["glitch and granular detail used as punctuation", "Glitch detail"],
  ],
  theoryPalette: [
    ["", "Leave instrumentation open"],
    ["Rhodes electric piano, muted electric guitar, rounded bass, pocket drums, and a restrained analog pad", "Rhodes · muted guitar · pocket rhythm"],
    ["grand piano, chamber strings, low brass, orchestral percussion, and a subtle synth foundation", "Piano · strings · brass · orchestral percussion"],
    ["analog polysynth, arpeggiator, electronic drums, sub bass, and selective vocal chops", "Analog synth · arpeggiator · electronic rhythm"],
    ["fingerpicked acoustic guitar, upright bass, brushed drums, and sparse harmonica", "Fingerpicked acoustic · upright bass · brushes"],
    ["two contrasting electric guitars, live bass, acoustic drums, and a low sustained keyboard layer", "Twin guitars · live rhythm section"],
    ["chamber strings, prepared piano, mallet percussion, and quiet found-sound texture", "Strings · prepared piano · mallets"],
    ["sample chops, dry drum machine hits, 808 low end, and a small detuned keyboard motif", "Sample chops · 808 · detuned keys"],
    ["broken drums, FM keys, reese bass, and chopped vocal texture", "Broken beat · FM keys · reese bass"],
    ["modular drones, granular field recordings, bowed metal, and sparse low percussion", "Modular drone · granular field texture"],
    ["nylon-string guitar, hand percussion, fretless bass, and breathy woodwind", "Nylon guitar · hand percussion · woodwind"],
    ["plucked heritage strings, bamboo flute, deep frame drums, and a modern electronic rhythm section", "Heritage strings · flute · hybrid rhythm"],
  ],
  theoryKey: [
    ["", "Leave key open"], ["C", "C"], ["C-sharp / D-flat", "C♯ / D♭"], ["D", "D"],
    ["E-flat", "E♭"], ["E", "E"], ["F", "F"], ["F-sharp / G-flat", "F♯ / G♭"],
    ["G", "G"], ["A-flat", "A♭"], ["A", "A"], ["B-flat", "B♭"], ["B", "B"],
  ],
  theoryMode: [
    ["", "Leave scale / mode open"],
    ["major", "Major"], ["natural minor", "Natural minor"], ["harmonic minor", "Harmonic minor"],
    ["melodic minor", "Melodic minor"], ["Dorian", "Dorian"], ["Phrygian", "Phrygian"],
    ["Lydian", "Lydian"], ["Mixolydian", "Mixolydian"], ["major pentatonic", "Major pentatonic"],
    ["minor pentatonic", "Minor pentatonic"], ["deliberately ambiguous modal center", "Modal / tonally ambiguous"],
  ],
  theoryTempo: [
    ["", "Leave tempo character open"],
    ["very slow and suspended", "Very slow · suspended"],
    ["slow with generous breathing room", "Slow · spacious"],
    ["mid-tempo and pocket-focused", "Mid-tempo · pocket"],
    ["upbeat with controlled forward motion", "Upbeat · controlled"],
    ["fast and driving", "Fast · driving"],
    ["fluid, with intentional push and pull rather than a rigid click", "Rubato · flexible"],
  ],
  theoryMeter: [
    ["", "Leave meter open"], ["4/4", "4/4 · common time"], ["3/4", "3/4 · triple meter"],
    ["6/8", "6/8 · compound duple"], ["12/8", "12/8 · compound quadruple"],
    ["5/4", "5/4 · asymmetric"], ["7/8", "7/8 · asymmetric"],
    ["alternating 4/4 and 3/4", "Alternating 4/4 + 3/4"], ["free time", "Free time"],
  ],
  theoryGroove: [
    ["", "Leave groove open"],
    ["a laid-back pocket that sits slightly behind the beat", "Laid-back · behind the beat"],
    ["a straight, precise pulse with restrained syncopation", "Straight · precise"],
    ["a swung pocket with elastic offbeats", "Swung · elastic"],
    ["a half-time backbeat with active subdivisions", "Half-time · active subdivisions"],
    ["a four-on-the-floor kick with syncopated upper percussion", "Four-on-the-floor"],
    ["a two-step garage pattern with clipped syncopation", "Two-step garage"],
    ["a broken-beat groove with displaced accents", "Broken beat"],
    ["a rolling compound-meter pulse", "Rolling compound meter"],
    ["an additive pattern whose accents make the odd meter feel singable", "Additive odd-meter accents"],
    ["a sparse pulse that periodically dissolves into free texture", "Pulse ↔ free texture"],
  ],
  theoryHarmony: [
    ["", "Leave harmonic language open"],
    ["clear functional harmony with purposeful cadences", "Functional · clear cadences"],
    ["extended seventh and ninth chords with smooth voice leading", "Extended 7ths / 9ths"],
    ["a modal vamp with slow internal voice movement", "Modal vamp"],
    ["suspended and quartal voicings that avoid obvious resolution", "Suspended · quartal"],
    ["a persistent pedal tone under changing upper harmony", "Pedal point"],
    ["selective modal interchange for section contrast", "Modal interchange"],
    ["chromatic-mediant movement used at major transitions", "Chromatic mediants"],
    ["blues-derived dominant harmony and call-and-response phrasing", "Blues language"],
    ["minimal harmonic motion with evolving orchestration", "Static harmony · evolving color"],
    ["dense contemporary-jazz color, but with a clearly singable top line", "Contemporary jazz density"],
  ],
  theoryVocal: [
    ["", "Leave vocal identity open"],
    ["instrumental, with the principal instrument carrying the melodic role", "Instrumental"],
    ["an intimate female alto, breath-led in verses and fuller in choruses", "Female alto · intimate → full"],
    ["a clear female mezzo-soprano with agile phrasing and restrained runs", "Female mezzo · agile"],
    ["a warm male baritone with close diction and controlled upper-register lift", "Male baritone · warm"],
    ["a light male tenor that moves from conversational delivery to open sustained notes", "Male tenor · conversational → open"],
    ["two contrasting lead voices that trade phrases and join only at structural peaks", "Duet · traded phrases"],
    ["spoken-word verses with a melodic sung refrain", "Spoken verse · sung refrain"],
    ["a small mixed choir used as an arrangement layer rather than a constant lead", "Mixed choir · selective"],
    ["a deliberately processed lead voice with human diction preserved beneath textural effects", "Processed lead · intelligible"],
  ],
  theoryArc: [
    ["", "Leave emotional arc open"],
    ["hushed and inward at the opening, gradually confident, then unresolved at the end", "Intimate → confident → unresolved"],
    ["tense and restrained, releasing fully only in the final chorus", "Restrained tension → final release"],
    ["serene at first, increasingly luminous, and quietly settled", "Serene → luminous → settled"],
    ["playful and kinetic, with a brief vulnerable center before the return", "Kinetic → vulnerable → return"],
    ["brooding, accumulative, and ultimately cathartic", "Brooding → accumulation → catharsis"],
    ["monumental at the outset, stripped to fragility, then rebuilt with greater scale", "Monumental → fragile → rebuilt"],
    ["nostalgic without sentimentality, ending with open space rather than a fade", "Nostalgic → open ending"],
    ["steady and meditative, changing through timbre rather than loudness", "Meditative · timbral evolution"],
  ],
  theoryForm: [
    ["", "Let the guide infer form"],
    ["Intro → Verse → Pre-Chorus → Chorus → Verse → Chorus → Bridge → Final Chorus → Outro", "Full vocal-song arc"],
    ["Intro → Verse → Chorus → Verse → Chorus → Instrumental → Final Chorus → Outro", "Direct verse / chorus"],
    ["Cold Open → Verse → Refrain → Verse → Refrain → Coda", "Refrain form"],
    ["Exposition → Development → Climax → Reprise → Coda", "Instrumental narrative"],
    ["Theme A → Theme B → Theme A variation → Breakdown → Combined return", "Contrasting themes"],
    ["Gradual introduction of layers → central peak → progressive subtraction", "Additive / subtractive arc"],
    ["Through-composed sections with no literal chorus repetition", "Through-composed"],
    ["Loop-based form with clearly audible four- or eight-bar mutations", "Loop with staged mutations"],
  ],
  theoryProduction: [
    ["", "Leave production open"],
    ["warm analog color, rounded low end, open dynamics, and wide but restrained ambience", "Warm analog · open dynamics"],
    ["close, dry, tactile foreground detail with very little artificial space", "Close · dry · tactile"],
    ["polished modern width, disciplined sub bass, clear vocal focus, and controlled transients", "Polished modern · controlled"],
    ["deep front-to-back space, long decays, and dark high-frequency balance", "Deep · dark · reverberant"],
    ["bright live-room energy with audible performance interaction and limited editing", "Live room · human"],
    ["cinematic scale with high dynamic contrast and carefully staged depth", "Cinematic · high contrast"],
    ["lo-fi patina used sparingly around a clean and intelligible center", "Selective lo-fi patina"],
    ["club-weight low end with mono-compatible bass and crisp spatial percussion", "Club low end · spatial percussion"],
    ["headphone-focused microdetail, subtle stereo motion, and an uncluttered center", "Headphone microdetail"],
  ],
  theoryScene: [
    ["", "Leave listening context open"],
    ["a solitary late-night headphone listen in a rain-lit city", "Late-night headphones"],
    ["a small live room where the listener can hear players react to one another", "Intimate live room"],
    ["a wide festival-scale final chorus without losing verse intimacy", "Intimate verse · festival chorus"],
    ["an opening-title sequence that must establish a world before dialogue begins", "Opening titles"],
    ["a focused night drive with forward motion but no aggressive brightness", "Night drive"],
    ["a gallery installation that rewards slow attention and seamless repetition", "Gallery / installation"],
    ["a physical club system where low-frequency clarity matters more than loudness", "Club system"],
  ],
};

const THEORY_HINTS = {
  Dorian: "Dorian keeps a minor center but raises scale degree six, which can create lift without turning fully major.",
  Phrygian: "Phrygian lowers scale degree two, giving the tonal center immediate friction.",
  Lydian: "Lydian raises scale degree four, often creating an open or weightless major color.",
  Mixolydian: "Mixolydian lowers scale degree seven, producing an open dominant color without a leading-tone pull.",
  "harmonic minor": "Harmonic minor raises scale degree seven, strengthening dominant-to-tonic motion.",
  "melodic minor": "Melodic minor supplies a minor tonic with raised sixth and seventh colors useful for modern jazz harmony.",
  "6/8": "6/8 is compound duple: two broad beats, each naturally divided into three.",
  "12/8": "12/8 supports four broad beats with triplet subdivision and a rolling pocket.",
  "5/4": "5/4 works best when its accent grouping is made audible, such as 3+2 or 2+3.",
  "7/8": "7/8 becomes musical rather than mathematical when the groove exposes a clear grouping such as 2+2+3.",
};

const THEORY_PRESETS = [
  {
    name: "Midnight neo-soul", note: "slow-bloom vocal record",
    duration: 150,
    fields: { theoryGenre: "alternative R&B with neo-soul harmony", theoryFusion: "restrained dream-pop texture", theoryPalette: "Rhodes electric piano, muted electric guitar, rounded bass, pocket drums, and a restrained analog pad", theoryKey: "B", theoryMode: "natural minor", theoryBpm: "76", theoryTempo: "slow with generous breathing room", theoryMeter: "4/4", theoryGroove: "a laid-back pocket that sits slightly behind the beat", theoryHarmony: "extended seventh and ninth chords with smooth voice leading", theoryVocal: "an intimate female alto, breath-led in verses and fuller in choruses", theoryArc: "hushed and inward at the opening, gradually confident, then unresolved at the end", theoryForm: "Intro → Verse → Pre-Chorus → Chorus → Verse → Chorus → Bridge → Final Chorus → Outro", theoryProduction: "warm analog color, rounded low end, open dynamics, and wide but restrained ambience", theoryScene: "a solitary late-night headphone listen in a rain-lit city" },
  },
  {
    name: "Orchestral slow burn", note: "narrative instrumental arc",
    duration: 180,
    fields: { theoryGenre: "cinematic orchestral music", theoryFusion: "minimalist repetition and gradual process", theoryPalette: "grand piano, chamber strings, low brass, orchestral percussion, and a subtle synth foundation", theoryKey: "D", theoryMode: "natural minor", theoryBpm: "72", theoryTempo: "slow with generous breathing room", theoryMeter: "6/8", theoryGroove: "a rolling compound-meter pulse", theoryHarmony: "a persistent pedal tone under changing upper harmony", theoryVocal: "instrumental, with the principal instrument carrying the melodic role", theoryArc: "monumental at the outset, stripped to fragility, then rebuilt with greater scale", theoryForm: "Exposition → Development → Climax → Reprise → Coda", theoryProduction: "cinematic scale with high dynamic contrast and carefully staged depth", theoryScene: "an opening-title sequence that must establish a world before dialogue begins" },
  },
  {
    name: "Two-step afterglow", note: "garage rhythm · alt-pop hook",
    duration: 135,
    fields: { theoryGenre: "UK garage with modern alt-pop songwriting", theoryFusion: "ambient negative space", theoryPalette: "broken drums, FM keys, reese bass, and chopped vocal texture", theoryKey: "F-sharp / G-flat", theoryMode: "Dorian", theoryBpm: "132", theoryTempo: "upbeat with controlled forward motion", theoryMeter: "4/4", theoryGroove: "a two-step garage pattern with clipped syncopation", theoryHarmony: "a modal vamp with slow internal voice movement", theoryVocal: "a clear female mezzo-soprano with agile phrasing and restrained runs", theoryArc: "playful and kinetic, with a brief vulnerable center before the return", theoryForm: "Intro → Verse → Chorus → Verse → Chorus → Instrumental → Final Chorus → Outro", theoryProduction: "club-weight low end with mono-compatible bass and crisp spatial percussion", theoryScene: "a focused night drive with forward motion but no aggressive brightness" },
  },
  {
    name: "Odd-meter post-rock", note: "singable 7/8 · long crescendo",
    duration: 210,
    fields: { theoryGenre: "alternative rock with post-rock dynamics", theoryFusion: "chamber-orchestral detail", theoryPalette: "two contrasting electric guitars, live bass, acoustic drums, and a low sustained keyboard layer", theoryKey: "D", theoryMode: "Dorian", theoryBpm: "98", theoryTempo: "mid-tempo and pocket-focused", theoryMeter: "7/8", theoryGroove: "an additive pattern whose accents make the odd meter feel singable", theoryHarmony: "suspended and quartal voicings that avoid obvious resolution", theoryVocal: "instrumental, with the principal instrument carrying the melodic role", theoryArc: "brooding, accumulative, and ultimately cathartic", theoryForm: "Theme A → Theme B → Theme A variation → Breakdown → Combined return", theoryProduction: "bright live-room energy with audible performance interaction and limited editing", theoryScene: "a small live room where the listener can hear players react to one another" },
  },
  {
    name: "Chamber mechanism", note: "5/4 · timbral evolution",
    duration: 165,
    fields: { theoryGenre: "modern chamber music with minimalist repetition", theoryPalette: "chamber strings, prepared piano, mallet percussion, and quiet found-sound texture", theoryKey: "A", theoryMode: "deliberately ambiguous modal center", theoryBpm: "84", theoryTempo: "mid-tempo and pocket-focused", theoryMeter: "5/4", theoryGroove: "an additive pattern whose accents make the odd meter feel singable", theoryHarmony: "minimal harmonic motion with evolving orchestration", theoryVocal: "instrumental, with the principal instrument carrying the melodic role", theoryArc: "steady and meditative, changing through timbre rather than loudness", theoryForm: "Gradual introduction of layers → central peak → progressive subtraction", theoryProduction: "close, dry, tactile foreground detail with very little artificial space", theoryScene: "a gallery installation that rewards slow attention and seamless repetition" },
  },
  {
    name: "Human folk lift", note: "6/8 · intimate ensemble",
    duration: 145,
    fields: { theoryGenre: "indie folk with contemporary acoustic production", theoryFusion: "gospel-informed harmony and backing vocals", theoryPalette: "fingerpicked acoustic guitar, upright bass, brushed drums, and sparse harmonica", theoryKey: "G", theoryMode: "major", theoryBpm: "78", theoryTempo: "slow with generous breathing room", theoryMeter: "6/8", theoryGroove: "a rolling compound-meter pulse", theoryHarmony: "clear functional harmony with purposeful cadences", theoryVocal: "a warm male baritone with close diction and controlled upper-register lift", theoryArc: "serene at first, increasingly luminous, and quietly settled", theoryForm: "Intro → Verse → Pre-Chorus → Chorus → Verse → Chorus → Bridge → Final Chorus → Outro", theoryProduction: "bright live-room energy with audible performance interaction and limited editing", theoryScene: "a small live room where the listener can hear players react to one another" },
  },
];

function setStatus(kind, text) {
  $("statusDot").className = `status-dot ${kind || ""}`;
  $("statusText").textContent = text;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = {};
  try { body = await response.json(); } catch (_) { /* empty response */ }
  if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`);
  return body;
}

async function refreshHealth() {
  try {
    const state = await api("/api/health");
    const complete = Object.values(state.models).every(Boolean);
    if (!state.engine) setStatus("error", "Inference engine unavailable");
    else if (!complete) setStatus("error", "Model files incomplete");
    else if (state.loaded) setStatus("loaded", "Models resident · local GPU");
    else setStatus("ready", "Engine ready · weights unloaded");
    $("loadModel").disabled = state.loaded || !complete;
    $("unloadModel").disabled = !state.loaded;
  } catch (error) {
    setStatus("error", error.message);
  }
}

function setBusy(button, busy, label) {
  if (!button.dataset.originalHtml) button.dataset.originalHtml = button.innerHTML;
  button.disabled = busy;
  button.innerHTML = busy ? label : button.dataset.originalHtml;
}

async function loadModels() {
  const button = $("loadModel");
  setBusy(button, true, "Loading…");
  setStatus("", "Loading FP16 DiT, INT8 encoder, and DAV…");
  try {
    await api("/api/load", { method: "POST" });
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(button, false, "");
    await refreshHealth();
  }
}

async function unloadModels() {
  const button = $("unloadModel");
  setBusy(button, true, "Unloading…");
  try {
    await api("/api/unload", { method: "POST" });
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(button, false, "");
    await refreshHealth();
  }
}

function requestBody() {
  return {
    global_metadata: $("globalMetadata").value,
    vocal_details: $("vocalDetails").value,
    arrangement: $("arrangement").value,
    lyrics: $("lyrics").value,
    duration: Number($("duration").value),
    seed: Number($("seed").value),
    steps: Number($("steps").value),
    cfg: Number($("cfg").value),
    top_k: Number($("topK").value),
    batch: Number($("batch").value),
    sampler: $("sampler").value,
    scheduler: $("scheduler").value,
    tiled_decode: $("tiledDecode").checked,
    tile_size: Number($("tileSize").value),
    tile_overlap: Number($("tileOverlap").value),
    output_format: $("outputFormat").value,
    quality: $("quality").value,
  };
}

function preparePerformance(request) {
  $("performanceMetadata").textContent = request.global_metadata || "—";
  $("performanceVocals").textContent = request.vocal_details || "Instrumental / unspecified";
  $("performanceArrangement").textContent = request.arrangement || "—";
  const container = $("performanceLyrics");
  container.replaceChildren();
  activeLyricIndex = -1;
  lyricRows = request.lyrics.split(/\r?\n/).filter((line) => line.trim()).map((line) => {
    const p = document.createElement("p");
    p.textContent = line;
    if (/^\s*\[.*\]\s*$/.test(line)) p.className = "tag";
    container.appendChild(p);
    return p;
  });
  if (lyricRows.length) {
    lyricRows[0].classList.add("active");
    activeLyricIndex = 0;
  }
}

async function generate(event) {
  event.preventDefault();
  const button = $("generate");
  const request = requestBody();
  preparePerformance(request);
  setBusy(button, true, "Queued…");
  $("renderStatus").textContent = "Submitting the official MiniMax graph…";
  try {
    let job = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    while (!["complete", "error", "cancelled"].includes(job.status)) {
      $("renderStatus").textContent = job.status === "waiting" ? "Waiting for the local generation lane…" : "Rendering locally. Long songs can take a while.";
      await sleep(1200);
      job = await api(`/api/jobs/${job.id}`);
    }
    if (job.status === "error") throw new Error(job.error || "Generation failed");
    if (job.status === "cancelled") throw new Error("Generation was cancelled");
    renderAudioOutputs(job.audio);
    $("renderStatus").textContent = `${job.audio.length} local audio file${job.audio.length === 1 ? "" : "s"} ready.`;
    await refreshHealth();
  } catch (error) {
    $("renderStatus").textContent = error.message;
  } finally {
    setBusy(button, false, "");
  }
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00";
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

function updateLyrics(audio) {
  if (!lyricRows.length || !audio.duration) return;
  const index = Math.min(lyricRows.length - 1, Math.floor((audio.currentTime / audio.duration) * lyricRows.length));
  if (index !== activeLyricIndex) {
    lyricRows.forEach((row, i) => row.classList.toggle("active", i === index));
    const container = $("performanceLyrics");
    const targetTop = lyricRows[index].offsetTop - (container.clientHeight / 2) + (lyricRows[index].clientHeight / 2);
    container.scrollTo({ top: Math.max(0, targetTop), behavior: "smooth" });
    activeLyricIndex = index;
  }
  $("playTime").textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
}

function connectVisualizer(audio) {
  if (!audioContext) audioContext = new AudioContext();
  if (audio.dataset.connected !== "true") {
    const source = audioContext.createMediaElementSource(audio);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = .82;
    source.connect(analyser);
    analyser.connect(audioContext.destination);
    audio.dataset.connected = "true";
  }
  currentAudio = audio;
  audioContext.resume();
  cancelAnimationFrame(animationFrame);
  drawSpectrum();
}

function drawSpectrum() {
  const canvas = $("spectrum");
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const values = new Uint8Array(analyser ? analyser.frequencyBinCount : 64);
  if (analyser && currentAudio && !currentAudio.paused) analyser.getByteFrequencyData(values);
  const bars = 72;
  const gap = 3;
  const barWidth = Math.max(2, (width - gap * (bars - 1)) / bars);
  for (let i = 0; i < bars; i += 1) {
    const sample = values[Math.floor((i / bars) * values.length)] || (3 + 5 * Math.sin(i * .7));
    const barHeight = Math.max(2, (sample / 255) * (height - 6));
    const x = i * (barWidth + gap);
    context.fillStyle = `rgba(244, 242, 235, ${.38 + (sample / 255) * .62})`;
    context.fillRect(x, height - barHeight, barWidth, barHeight);
  }
  if (currentAudio) updateLyrics(currentAudio);
  animationFrame = requestAnimationFrame(drawSpectrum);
}

function renderAudioOutputs(urls) {
  const container = $("audioOutputs");
  container.replaceChildren();
  urls.forEach((url, index) => {
    const card = document.createElement("div");
    card.className = "audio-card";
    const label = document.createElement("span");
    label.textContent = `Take ${String(index + 1).padStart(2, "0")}`;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = url;
    audio.addEventListener("play", () => {
      document.querySelectorAll("audio").forEach((other) => { if (other !== audio) other.pause(); });
      $("nowPlaying").textContent = `TAKE ${String(index + 1).padStart(2, "0")} · MINIMAX MUSIC 3`;
      connectVisualizer(audio);
    });
    const download = document.createElement("a");
    download.href = url;
    download.download = "";
    download.textContent = "Download ↘";
    card.append(label, audio, download);
    container.appendChild(card);
  });
}

function randomSeedFor(inputId) {
  const values = new Uint32Array(2);
  crypto.getRandomValues(values);
  $(inputId).value = String((values[0] * 0x100000 + (values[1] & 0xfffff)) % Number.MAX_SAFE_INTEGER);
}

function randomSeed() {
  randomSeedFor("seed");
}

async function interrupt() {
  try {
    await api("/api/interrupt", { method: "POST" });
    $("renderStatus").textContent = "Interrupt requested.";
  } catch (error) {
    $("renderStatus").textContent = error.message;
  }
}

function switchWorkspace(panelId) {
  document.querySelectorAll(".workspace-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === panelId);
  });
  document.querySelectorAll(".workspace-panel").forEach((panel) => {
    const active = panel.id === panelId;
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function theoryValue(id) {
  return $(id).value.trim();
}

function withIndefiniteArticle(value) {
  return `${/^[aeiou]/i.test(value) ? "an" : "a"} ${value}`;
}

function buildTheoryPrompt() {
  const genre = theoryValue("theoryGenre");
  const fusion = theoryValue("theoryFusion");
  const palette = theoryValue("theoryPalette");
  const key = theoryValue("theoryKey");
  const mode = theoryValue("theoryMode");
  const bpm = theoryValue("theoryBpm");
  const tempo = theoryValue("theoryTempo");
  const meter = theoryValue("theoryMeter");
  const groove = theoryValue("theoryGroove");
  const harmony = theoryValue("theoryHarmony");
  const vocal = theoryValue("theoryVocal");
  const arc = theoryValue("theoryArc");
  const form = theoryValue("theoryForm");
  const production = theoryValue("theoryProduction");
  const scene = theoryValue("theoryScene");
  const custom = theoryValue("theoryCustom");
  const sentences = [];

  if (genre && fusion) sentences.push(`Create ${withIndefiniteArticle(genre)} piece, colored by ${fusion}.`);
  else if (genre) sentences.push(`Create ${withIndefiniteArticle(genre)} piece.`);
  else if (fusion) sentences.push(`Use ${fusion} as a secondary color without forcing a primary genre.`);

  if (bpm && tempo) sentences.push(`Set the requested tempo at ${bpm} BPM, with motion that feels ${tempo}.`);
  else if (bpm) sentences.push(`Set the requested tempo at ${bpm} BPM.`);
  else if (tempo) sentences.push(`Keep the tempo ${tempo}, without inventing an exact BPM.`);

  if (key && mode === "deliberately ambiguous modal center") {
    sentences.push(`Use ${key} as a loose tonal center while keeping the mode deliberately ambiguous.`);
  } else if (key && mode) {
    sentences.push(`Center the harmony on ${key} ${mode}.`);
  } else if (key) {
    sentences.push(`Use ${key} as the requested tonal center without forcing a particular scale.`);
  } else if (mode) {
    sentences.push(`Favor a ${mode} color without forcing a specific key.`);
  }

  if (meter && groove) sentences.push(`Write in ${meter}, shaped by ${groove}.`);
  else if (meter) sentences.push(`Use ${meter} as the requested meter.`);
  else if (groove) sentences.push(`Shape the rhythm around ${groove}.`);
  if (harmony) sentences.push(`Use ${harmony}.`);
  if (palette) sentences.push(`Build the core palette from ${palette}.`);
  if (vocal) sentences.push(`Lead performance: ${vocal}.`);
  if (arc) sentences.push(`Emotional trajectory: ${arc}.`);
  if (form) sentences.push(`Suggested form: ${form}.`);
  if (production) sentences.push(`Production profile: ${production}.`);
  if (scene) sentences.push(`Listening context: ${scene}.`);
  if (custom) sentences.push(`Hard direction: ${custom.replace(/[\s.]+$/, "")}.`);
  return sentences.join(" ");
}

function updateTheoryHint() {
  const hints = [];
  const mode = theoryValue("theoryMode");
  const meter = theoryValue("theoryMeter");
  if (THEORY_HINTS[mode]) hints.push(THEORY_HINTS[mode]);
  if (THEORY_HINTS[meter]) hints.push(THEORY_HINTS[meter]);
  if (!hints.length) hints.push("Leave a field open when it is not a real requirement; the prompt rewriter will not fabricate it.");
  hints.push("MiniMax follows musical descriptions generatively, so exact symbolic values may still vary in the rendered audio.");
  $("theoryHint").textContent = hints.join(" ");
}

function updateTheoryPreview() {
  const preview = $("theoryPromptPreview");
  preview.value = buildTheoryPrompt();
  preview.placeholder = "Choose a recipe or a few intentional details. Unselected dimensions stay open.";
  updateTheoryHint();
}

function clearActiveTheoryPreset() {
  document.querySelectorAll(".theory-preset.active").forEach((button) => button.classList.remove("active"));
}

function applyTheoryPreset(preset, button) {
  Object.entries(preset.fields).forEach(([id, value]) => { $(id).value = value; });
  if (preset.duration) $("guideDuration").value = String(preset.duration);
  clearActiveTheoryPreset();
  button.classList.add("active");
  updateTheoryPreview();
}

function renderTheoryPresets() {
  const container = $("theoryPresets");
  container.replaceChildren();
  THEORY_PRESETS.forEach((preset) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "theory-preset";
    const name = document.createElement("strong");
    name.textContent = preset.name;
    const note = document.createElement("small");
    note.textContent = preset.note;
    button.append(name, note);
    button.addEventListener("click", () => applyTheoryPreset(preset, button));
    container.appendChild(button);
  });
}

function populateTheoryOptions() {
  Object.entries(THEORY_OPTIONS).forEach(([id, options]) => {
    const select = $(id);
    select.replaceChildren();
    options.forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    });
  });
}

function theoryBriefOrStatus() {
  const text = buildTheoryPrompt();
  if (!text) $("theoryHint").textContent = "Choose at least one intentional detail or start from a recipe.";
  return text;
}

function flashButton(button, text) {
  const original = button.textContent;
  button.textContent = text;
  window.setTimeout(() => { button.textContent = original; }, 1200);
}

function appendTheoryBrief(replace = false) {
  const text = theoryBriefOrStatus();
  if (!text) return;
  const direction = $("guideDirection");
  if (replace) {
    direction.value = text;
  } else {
    const block = `Guided production brief:\n${text}`;
    const current = direction.value.trim();
    const marker = /\n{0,2}Guided production brief:\n[\s\S]*$/;
    direction.value = marker.test(current)
      ? current.replace(marker, `\n\n${block}`)
      : `${current}${current ? "\n\n" : ""}${block}`;
  }
  direction.dispatchEvent(new Event("input", { bubbles: true }));
  direction.focus({ preventScroll: true });
  direction.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function writeClipboard(text, button) {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const helper = document.createElement("textarea");
    helper.value = text;
    helper.style.position = "fixed";
    helper.style.opacity = "0";
    document.body.appendChild(helper);
    helper.select();
    document.execCommand("copy");
    helper.remove();
  }
  flashButton(button, "Copied");
}

function clearTheoryBrief() {
  Object.keys(THEORY_OPTIONS).forEach((id) => { $(id).value = ""; });
  $("theoryBpm").value = "";
  $("theoryCustom").value = "";
  clearActiveTheoryPreset();
  updateTheoryPreview();
}

function insertSectionTag(tag) {
  const field = $("guideLyrics");
  const start = field.selectionStart;
  const end = field.selectionEnd;
  const before = field.value.slice(0, start);
  const after = field.value.slice(end);
  const prefix = before && !before.endsWith("\n") ? "\n" : "";
  const suffix = after.startsWith("\n") || !after ? "\n" : "\n\n";
  const insertion = `${prefix}${tag}${suffix}`;
  field.setRangeText(insertion, start, end, "end");
  field.focus();
}

function initializeTheoryLab() {
  populateTheoryOptions();
  renderTheoryPresets();
  document.querySelectorAll(".theory-selector-grid select, #theoryBpm, #theoryCustom").forEach((control) => {
    control.addEventListener("input", () => { clearActiveTheoryPreset(); updateTheoryPreview(); });
  });
  document.querySelectorAll("[data-section-tag]").forEach((button) => {
    button.addEventListener("click", () => insertSectionTag(button.dataset.sectionTag));
  });
  $("appendTheoryBrief").addEventListener("click", () => appendTheoryBrief(false));
  $("replaceTheoryBrief").addEventListener("click", () => appendTheoryBrief(true));
  $("copyTheoryBrief").addEventListener("click", () => writeClipboard(theoryBriefOrStatus(), $("copyTheoryBrief")));
  $("clearTheoryBrief").addEventListener("click", clearTheoryBrief);
  updateTheoryPreview();
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "unknown size";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit > 2 ? 2 : 1)} ${units[unit]}`;
}

function renderGuideModels(filter = "") {
  const list = $("guideModelList");
  list.replaceChildren();
  const needle = filter.trim().toLowerCase();
  const matches = guideModels.filter((model) => model.path.toLowerCase().includes(needle));
  if (!matches.length) {
    const empty = document.createElement("p");
    empty.className = "empty-model-list";
    empty.textContent = "No local .safetensors checkpoints match this filter.";
    list.appendChild(empty);
    return;
  }
  matches.forEach((model) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-option";
    const copy = document.createElement("span");
    const name = document.createElement("b");
    name.textContent = model.name;
    const path = document.createElement("small");
    path.textContent = model.path;
    copy.append(name, path);
    const size = document.createElement("em");
    size.textContent = formatBytes(model.bytes);
    button.append(copy, size);
    button.addEventListener("click", () => {
      $("guideModel").value = model.path;
      $("guideModelDialog").close();
    });
    list.appendChild(button);
  });
}

async function refreshGuideModels() {
  try {
    const state = await api("/api/guide/models");
    guideModels = state.models || [];
    $("guideModelRoot").textContent = state.model_root;
    const current = $("guideModel").value;
    if (!current || (!guideModels.some((model) => model.path === current) && state.default_model)) {
      $("guideModel").value = state.default_model;
    }
    renderGuideModels($("guideModelSearch").value);
  } catch (error) {
    guideModels = [];
    $("guideModelRoot").textContent = error.message;
    renderGuideModels();
  }
}

async function refreshGuideStatus() {
  try {
    const state = await api("/api/guide/status");
    const enabled = Boolean(state.loaded);
    $("guideEnabled").checked = enabled;
    $("guideFields").disabled = !enabled;
    $("browseGuideModels").disabled = enabled;
    const heading = document.querySelector(".guide-runtime-card h2");
    if (enabled) {
      heading.textContent = "Prompt Guide is resident";
      $("guideStatus").textContent = `${state.loaded_model} · isolated local GPU lane`;
    } else if (state.running) {
      heading.textContent = "Prompt Guide is starting";
      $("guideStatus").textContent = "The private process is active; weights are not yet confirmed resident.";
    } else if (!state.default_exists) {
      heading.textContent = "Prompt Guide is off";
      $("guideStatus").textContent = "Default checkpoint is missing. Browse another local Krea2-compatible .safetensors file.";
    } else {
      heading.textContent = "Prompt Guide is off";
      $("guideStatus").textContent = "No secondary process and no Qwen weights are resident.";
    }
  } catch (error) {
    $("guideEnabled").checked = false;
    $("guideFields").disabled = true;
    document.querySelector(".guide-runtime-card h2").textContent = "Prompt Guide unavailable";
    $("guideStatus").textContent = error.message;
  }
}

async function toggleGuide() {
  const toggle = $("guideEnabled");
  const requested = toggle.checked;
  toggle.disabled = true;
  $("guideStatus").textContent = requested
    ? "Starting the isolated Comfy lane and loading Qwen as Krea2…"
    : "Interrupting the guide, releasing weights, and stopping its private process…";
  try {
    if (requested) {
      await api("/api/guide/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: $("guideModel").value }),
      });
    } else {
      await api("/api/guide/unload", { method: "POST" });
    }
  } catch (error) {
    $("guideStatus").textContent = error.message;
  } finally {
    toggle.disabled = false;
    await refreshGuideStatus();
  }
}

function promptGuideBody() {
  return {
    model: $("guideModel").value,
    direction: $("guideDirection").value,
    lyrics: $("guideLyrics").value,
    constraints: $("guideConstraints").value,
    duration: Number($("guideDuration").value),
    steps: Number($("guideSteps").value),
    cfg: Number($("guideCfg").value),
    acoustic_top_k: Number($("guideAcousticTopK").value),
    sampler: $("guideSampler").value,
    scheduler: $("guideScheduler").value,
    tiled_decode: $("guideTiled").checked,
    max_length: Number($("guideMaxLength").value),
    temperature: Number($("guideTemperature").value),
    top_k: Number($("guideTopK").value),
    top_p: Number($("guideTopP").value),
    min_p: Number($("guideMinP").value),
    repetition_penalty: Number($("guideRepeatPenalty").value),
    seed: Number($("guideSeed").value),
    presence_penalty: Number($("guidePresencePenalty").value),
    sampling: $("guideSampling").checked,
    thinking: $("guideThinking").checked,
    use_default_template: $("guideDefaultTemplate").checked,
  };
}

function renderGuideResult(result) {
  const sections = result.sections || {};
  $("guideGlobalOutput").textContent = sections["Global Metadata"] || "Section was not isolated; inspect the raw response below.";
  $("guideVocalOutput").textContent = sections["Vocal Details"] || "Section was not isolated; inspect the raw response below.";
  $("guideArrangementOutput").textContent = sections.Arrangement || "Section was not isolated; inspect the raw response below.";
  $("guideTuningOutput").textContent = sections["Tuning Notes"] || "Section was not isolated; inspect the raw response below.";
  $("guideRawOutput").textContent = result.text || "";
  $("guideResults").hidden = false;
  $("guideResults").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runPromptGuide(event) {
  event.preventDefault();
  const button = $("runPromptGuide");
  setBusy(button, true, "Rewriting…");
  $("guideRunStatus").textContent = "Qwen is building a MiniMax-specific structured caption locally…";
  try {
    const result = await api("/api/guide/enhance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(promptGuideBody()),
    });
    renderGuideResult(result);
    $("guideRunStatus").textContent = "Finished locally. Copy only the sections you want.";
  } catch (error) {
    $("guideRunStatus").textContent = error.message;
  } finally {
    setBusy(button, false, "");
    await refreshGuideStatus();
  }
}

async function copyGuideText(elementId, button) {
  const text = $(elementId).textContent;
  await writeClipboard(text, button);
}

document.querySelectorAll("[data-port]").forEach((link) => {
  link.href = `${location.protocol}//${location.hostname}:${link.dataset.port}/`;
});
document.querySelectorAll(".workspace-tab").forEach((button) => {
  button.addEventListener("click", () => switchWorkspace(button.dataset.tab));
});
document.querySelectorAll(".copy-guide").forEach((button) => {
  button.addEventListener("click", () => copyGuideText(button.dataset.copy, button));
});
$("duration").addEventListener("input", () => { $("durationReadout").textContent = `${$("duration").value} s`; });
$("randomSeed").addEventListener("click", randomSeed);
$("guideRandomSeed").addEventListener("click", () => randomSeedFor("guideSeed"));
$("refreshStatus").addEventListener("click", refreshHealth);
$("loadModel").addEventListener("click", loadModels);
$("unloadModel").addEventListener("click", unloadModels);
$("interrupt").addEventListener("click", interrupt);
$("composer").addEventListener("submit", generate);
$("guideEnabled").addEventListener("change", toggleGuide);
$("browseGuideModels").addEventListener("click", async () => {
  await refreshGuideModels();
  $("guideModelDialog").showModal();
  $("guideModelSearch").focus();
});
$("guideModelSearch").addEventListener("input", () => renderGuideModels($("guideModelSearch").value));
$("promptGuideForm").addEventListener("submit", runPromptGuide);

initializeTheoryLab();
drawSpectrum();
refreshHealth();
refreshGuideModels();
refreshGuideStatus();
setInterval(() => { refreshHealth(); refreshGuideStatus(); }, 15000);
