import tailwindcss from "@tailwindcss/vite";
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

// Usage: VITE_API_TARGET=http://192.168.0.100:52415 npm run dev
export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  server: {
    proxy: Object.fromEntries(
      [
        "/v1",
        "/state",
        "/models",
        "/instance",
        "/place_instance",
        "/node_id",
        "/onboarding",
        "/download",
        "/health",
      ].map((route) => [
        route,
        process.env.VITE_API_TARGET || "http://localhost:52415",
      ]),
    ),
  },
});
