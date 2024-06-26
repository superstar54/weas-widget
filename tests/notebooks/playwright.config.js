const baseConfig = require('@jupyterlab/galata/lib/playwright-config');

module.exports = {
  ...baseConfig,
  timeout: 240000,
  workers: process.env.CI ? 1 : 1,
  webServer: {
    command: 'jlpm start',
    url: 'http://localhost:8888/lab',
    timeout: 120 * 1000,
    reuseExistingServer: !process.env.CI,
  },
  retries: 0,
};
