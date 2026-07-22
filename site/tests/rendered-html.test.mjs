import assert from "node:assert/strict";
import { access } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), {
    ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
  }, { waitUntil() {}, passThroughOnException() {} });
}

test("renders the complete match-intelligence story", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>Match Intelligence 2026 \| Argentina vs Switzerland<\/title>/i);
  assert.match(html, /Before the whistle/);
  assert.match(html, /3,961/);
  assert.match(html, /The honest limitation/i);
  assert.match(html, /finale-web\.mp4/);
  assert.match(html, /final-paper\.pdf/);
});

test("ships all presentation assets", async () => {
  await Promise.all(["hero.jpg", "vision.jpg", "og.png", "finale-web.mp4", "final-paper.pdf"].map((file) => access(new URL(`../public/${file}`, import.meta.url))));
});
