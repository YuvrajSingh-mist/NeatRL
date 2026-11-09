#!/usr/bin/env node
// Developer entry point so `npm run launch` works from the NeatRL repo root.
//
// The website lives in its own repo. It is gitignored here, so a checkout can
// sit in either place:
//   NeatRL/NeatRL-Website      (the layout .gitignore implies)
//   NeatRL/../NeatRL-Website   (what is on disk today)
// Pick whichever actually has a package.json, then forward the script name.
//
// The NeatRL library itself is pure Python - this file only wraps the docs site
// and the in-browser playground, and is never imported by the package.

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CANDIDATES = ["NeatRL-Website", "../NeatRL-Website"];

const script = process.argv[2];
const extra = process.argv.slice(3);

if (!script) {
  console.error("usage: node scripts/site.mjs <launch|serve|routes|og> [...args]");
  process.exit(2);
}

const site = CANDIDATES.map((rel) => resolve(root, rel)).find((dir) =>
  existsSync(resolve(dir, "package.json")),
);

if (!site) {
  console.error(
    `Could not find the website checkout. Looked for package.json in:\n` +
      CANDIDATES.map((rel) => `  ${resolve(root, rel)}`).join("\n") +
      `\n\nClone it (git clone <neatrl-website> NeatRL-Website) or set it up beside this repo.`,
  );
  process.exit(1);
}

const args = ["run", script, ...(extra.length ? ["--", ...extra] : [])];
const child = spawn("npm", args, {
  cwd: site,
  stdio: "inherit",
  // npm's shim on Windows is a .cmd, which needs a shell to launch.
  shell: process.platform === "win32",
});

child.on("error", (err) => {
  console.error(`Failed to start npm: ${err.message}`);
  process.exit(1);
});

// Propagate Ctrl+C so the Python dev server shuts down with the wrapper.
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 0);
});
