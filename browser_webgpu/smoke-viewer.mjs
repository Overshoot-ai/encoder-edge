import { chromium } from "playwright";

const url = process.env.VIEWER_SMOKE_URL || "http://127.0.0.1:3000";
const browser = await chromium.launch({
  channel: "chromium",
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
  ],
});

try {
  const context = await browser.newContext({ permissions: ["camera"] });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.stack || String(error)));
  await page.goto(url);
  await page.waitForFunction(() => {
    const match = document.querySelector("#frame-count")?.textContent.match(/\d+/);
    return match && Number(match[0]) >= 3;
  }, null, { timeout: 10 * 60 * 1000 });
  await page.getByRole("tab", { name: "264 raw vectors" }).click();
  await page.getByLabel("All 264 raw embedding vectors").click({ position: { x: 20, y: 20 } });
  await page.getByText("d767", { exact: true }).scrollIntoViewIfNeeded();
  const selectedToken = await page.locator("#token-title").textContent();
  const before = Number((await page.locator("#frame-count").textContent()).match(/\d+/)[0]);
  await page.waitForFunction((count) => {
    const match = document.querySelector("#frame-count")?.textContent.match(/\d+/);
    return match && Number(match[0]) > count;
  }, before, { timeout: 10 * 60 * 1000 });
  await page.getByText(selectedToken, { exact: true }).waitFor();
  const dimensions = await page.locator("#token-values label").count();
  await page.getByRole("button", { name: "Pause" }).click();
  await page.getByRole("button", { name: "Resume" }).waitFor();
  await page.waitForTimeout(1300);
  const pausedAt = Number((await page.locator("#frame-count").textContent()).match(/\d+/)[0]);
  await page.waitForTimeout(1300);
  const finalCount = Number((await page.locator("#frame-count").textContent()).match(/\d+/)[0]);
  const diagnostics = await page.evaluate(() => window.__gemmaEmbeddingDiagnostics);
  if (errors.length || dimensions !== 768 || pausedAt !== finalCount) {
    throw new Error(JSON.stringify({ errors, dimensions, pausedAt, finalCount }, null, 2));
  }
  console.log(JSON.stringify({ selectedToken, dimensions, pausedAt, finalCount, diagnostics }, null, 2));
} finally {
  await browser.close();
}
