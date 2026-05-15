import { expect, test } from "bun:test";

test("Bun.TOML.parse with non-string input throws", () => {
  expect(() => Bun.TOML.parse(SharedArrayBuffer as any)).toThrow();
  expect(() => Bun.TOML.parse(undefined as any)).toThrow();
  expect(() => Bun.TOML.parse(null as any)).toThrow();
});

// https://github.com/oven-sh/bun/issues/30825
// `\u{...}` escape with enough hex digits to overflow i64 used to panic
// the debug lexer. It must not crash.
test("Bun.TOML.parse doesn't crash on out-of-range \\u{} escape", () => {
  expect(() => Bun.TOML.parse(`key = "\\u{3333333316aaaaaaa}"`)).not.toThrow();
});
