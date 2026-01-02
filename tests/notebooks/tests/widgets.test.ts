// Modified from ipywidgets unit tests.


import { test } from '@jupyterlab/galata';

import { expect } from '@playwright/test';

import * as path from 'path';

test.describe('Widget Visual Regression', () => {
  test.beforeEach(async ({ page, tmpPath }) => {
    await page.contents.uploadDirectory(
      path.resolve(__dirname, './notebooks'),
      tmpPath
    );
    await page.filebrowser.openDirectory(tmpPath);
  });

  test('Run notebook widgets.ipynb and capture cell outputs', async ({
    page,
    tmpPath,
  }) => {
    const notebook = 'widgets.ipynb';
    await page.notebook.openByPath(`${tmpPath}/${notebook}`);
    await page.notebook.activate(notebook);

    const captures = new Array<Buffer>();
    const cellCount = await page.notebook.getCellCount();

    await page.notebook.runCellByCell({
      onAfterCellRun: async (cellIndex: number) => {
        let cell = await page.notebook.getCellOutput(cellIndex);
        const startTime = Date.now();
        const timeout = 60000; // Timeout in milliseconds, adjust as necessary

        // Polling for cell output to be not null
        while (!cell && Date.now() - startTime < timeout) {
          await page.waitForTimeout(3000); // Wait for 3 second before retrying
          console.log("waiting for cell output to be not null: ", cellIndex);
          cell = await page.notebook.getCellOutput(cellIndex);
        }

        if (cell) {
          captures.push(await cell.screenshot());
        } else {
          console.log("Cell output is not available for cell:", cellIndex);
        }
      },
    });

    await page.notebook.save();

    console.log("Cell count:", cellCount);
    console.log("Captures array length:", captures.length);
    captures.forEach((capture, index) => {
      console.log(`Capture[${index}]:`, capture);
    });

    for (let i = 0; i < cellCount; i++) {
      const image = `widgets-cell-${i}.png`;
      expect.soft(captures[i]).toMatchSnapshot(image);
    }
  });
});
