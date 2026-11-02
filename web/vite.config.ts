import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// The demo backend runs on :8000 (ANBAN_DEMO=1 python -m api.app); proxying
// /api keeps the browser on one origin, so no CORS setup is needed.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: loadEnv(mode, ".", "ANBAN_").ANBAN_API_TARGET || "http://127.0.0.1:8000",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
}));
