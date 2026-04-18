import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      // REMOVED: /api/:path* rewrite to avoid conflict with manual API routes
      {
        source: "/docs",
        destination: `${backendUrl}/docs`,
      }
    ];
  },
};

export default nextConfig;
