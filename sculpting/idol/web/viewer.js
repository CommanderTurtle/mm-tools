import * as THREE from "three";
import { Viewer, RenderMode, SceneFormat, SceneRevealMode } from "splats";

export class SplatPanel {
  constructor(root, message) {
    this.root = root;
    this.message = message;
    this.sequence = 0;
    this.current = null;
    this.cleanup = Promise.resolve();
  }

  async clear() {
    ++this.sequence;
    await this.disposeCurrent();
  }

  async disposeCurrent() {
    const item = this.current;
    this.current = null;
    if (item) {
      item.viewer.stop();
      item.resize.disconnect();
      this.cleanup = item.viewer
        .dispose()
        .catch(() => {})
        .finally(() => {
          item.renderer.dispose();
          item.renderer.forceContextLoss();
          item.renderer.domElement.remove();
        });
    }
    await this.cleanup;
  }

  async load(url) {
    const sequence = ++this.sequence;
    await this.disposeCurrent();
    if (sequence !== this.sequence) return;
    this.message.textContent = "Loading splats";
    this.message.hidden = false;
    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      precision: "highp",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0c0c0c, 1);
    renderer.setSize(this.root.clientWidth, this.root.clientHeight);
    renderer.domElement.tabIndex = 0;
    this.root.append(renderer.domElement);
    let viewer;
    try {
      // External ownership keeps upstream disposal from removing the studio's panel.
      viewer = new Viewer({
        rootElement: this.root,
        renderer,
        cameraUp: [0, -1, 0],
        initialCameraPosition: [0, -0.15, 2.73],
        initialCameraLookAt: [0, 0, 0],
        renderMode: RenderMode.OnChange,
        sharedMemoryForWorkers: false,
        sphericalHarmonicsDegree: 0,
        sceneRevealMode: matchMedia("(prefers-reduced-motion: reduce)").matches
          ? SceneRevealMode.Instant
          : SceneRevealMode.Default,
      });
      // Viewer shortcuts stay in the canvas, leaving form arrows and typing alone.
      window.removeEventListener("keydown", viewer.keyDownListener);
      renderer.domElement.addEventListener("keydown", viewer.keyDownListener);
      for (const controls of [
        viewer.perspectiveControls,
        viewer.orthographicControls,
      ]) {
        controls?.stopListenToKeyEvents();
        controls?.listenToKeyEvents(renderer.domElement);
      }
      const resize = new ResizeObserver(() => {
        renderer.setSize(
          Math.max(1, this.root.clientWidth),
          Math.max(1, this.root.clientHeight),
        );
        viewer.forceRenderNextFrame();
      });
      resize.observe(this.root);
      this.current = { viewer, renderer, resize, ready: false };
      // This library's thenable omits rejection forwarding; await its native promise.
      await viewer.addSplatScene(url, {
        format: SceneFormat.Ply,
        showLoadingUI: false,
        splatAlphaRemovalThreshold: 1,
        progressiveLoad: false,
      }).promise;
      if (sequence !== this.sequence) return;
      this.current.ready = true;
      this.message.hidden = true;
      if (!document.hidden) viewer.start();
    } catch (error) {
      if (sequence !== this.sequence) return;
      if (this.current) await this.disposeCurrent();
      else {
        renderer.dispose();
        renderer.forceContextLoss();
        renderer.domElement.remove();
      }
      if (sequence !== this.sequence) return;
      this.message.textContent = `Viewer: ${error.message}. The PLY download remains available.`;
      this.message.hidden = false;
    }
  }

  visibility(hidden = document.hidden) {
    const item = this.current;
    if (!item?.ready) return;
    if (hidden) item.viewer.stop();
    else if (!item.viewer.selfDrivenModeRunning) item.viewer.start();
  }

  reset() {
    const item = this.current;
    if (!item?.ready) return;
    item.viewer.controls.reset();
    item.viewer.forceRenderNextFrame();
  }
}
