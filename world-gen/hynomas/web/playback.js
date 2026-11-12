export class Playback {
  constructor(video) {
    this.video = video;
    this.reset();
  }
  reset() {
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = null;
    this.media = null;
    this.buffer = null;
    this.sequence = -1;
    this.init = null;
    this.failed = false;
  }
  async event(target, success, action) {
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timer);
        target.removeEventListener(success, ok);
        target.removeEventListener("error", fail);
      };
      const ok = () => {
        cleanup();
        resolve();
      };
      const fail = () => {
        cleanup();
        reject(Error("Video buffer unavailable"));
      };
      const timer = setTimeout(fail, 15000);
      target.addEventListener(success, ok, { once: true });
      target.addEventListener("error", fail, { once: true });
      try {
        action();
      } catch (error) {
        cleanup();
        reject(error);
      }
    });
  }
  async bytes(url) {
    const response = await fetch(url);
    if (!response.ok) throw Error("Video fragment unavailable");
    return response.arrayBuffer();
  }
  async append(result) {
    const v = this.video,
      wasPlaying = !v.paused,
      atEnd =
        v.ended || v.currentTime >= (result.playback_start_seconds || 0) - 0.25;
    if (
      !this.failed &&
      result.stream_init_url &&
      result.stream_segment_url &&
      typeof MediaSource !== "undefined" &&
      MediaSource.isTypeSupported(result.stream_mime_type || "")
    ) {
      try {
        if (result.stream_sequence === 0) {
          this.reset();
          this.media = new MediaSource();
          this.url = URL.createObjectURL(this.media);
          await this.event(this.media, "sourceopen", () => {
            v.src = this.url;
          });
          this.buffer = this.media.addSourceBuffer(result.stream_mime_type);
          this.buffer.mode = "sequence";
          this.init = result.stream_init_url;
          const init = await this.bytes(this.init);
          await this.event(this.buffer, "updateend", () =>
            this.buffer.appendBuffer(init),
          );
        }
        if (
          !this.buffer ||
          this.init !== result.stream_init_url ||
          result.stream_sequence !== this.sequence + 1
        )
          throw Error("Noncontiguous playback");
        const replace = result.stream_replace_start_seconds;
        if (Number.isFinite(replace)) {
          const end = this.buffer.buffered.length
            ? this.buffer.buffered.end(this.buffer.buffered.length - 1)
            : 0;
          if (end > replace) {
            this.buffer.abort();
            await this.event(this.buffer, "updateend", () =>
              this.buffer.remove(Math.max(0, replace), end),
            );
          }
          this.buffer.timestampOffset = Math.max(0, replace);
        }
        const bytes = await this.bytes(result.stream_segment_url);
        await this.event(this.buffer, "updateend", () =>
          this.buffer.appendBuffer(bytes),
        );
        this.sequence = result.stream_sequence;
        this.media.duration = result.frames / result.fps;
        if (atEnd)
          v.currentTime = Math.min(
            result.playback_start_seconds || 0,
            this.media.duration,
          );
        if (wasPlaying || atEnd) v.play().catch(() => {});
        return;
      } catch {
        this.failed = true;
      }
    }
    const time = atEnd ? result.playback_start_seconds || 0 : v.currentTime;
    await this.event(v, "loadedmetadata", () => {
      v.src = result.video_url;
    });
    v.currentTime = Math.min(time, v.duration || 0);
    if (wasPlaying || atEnd) v.play().catch(() => {});
  }
  async show(url) {
    this.reset();
    this.failed = true;
    await this.event(this.video, "loadedmetadata", () => {
      this.video.src = url;
    });
  }
}
