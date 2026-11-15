import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  // Rust sources and build output are rebuilt by cargo, not Vite. Watching them crashed `tauri dev`
  // with EBUSY on a DLL cargo was writing (target/debug/build/.../raptor_lib.dll).
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/target/**", "**/src-tauri/**"] } },
  envPrefix: ["VITE_", "TAURI_ENV_*"],
});
