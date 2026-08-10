/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  // 限制 build 期 ESLint 仅扫描应用代码,避开 e2e/ 下 Playwright 测试
  // (Playwright 是 devDependency,生产镜像内不会安装,导致 type check 找不到模块)
  eslint: {
    dirs: ['src', 'app'],
  },
  // 同上,把 e2e 排除出 type check 防止 next build 在生产镜像里失败
  typescript: {
    // tsconfig.json 里已配置 exclude,这里保留 ignoreBuildErrors=false 让真实代码错误依然能拦截
    ignoreBuildErrors: false,
  },
};

export default nextConfig;
