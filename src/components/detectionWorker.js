//@ts-nocheck
/* eslint-disable */
self.onmessage = async (e) => {
    try {
        console.log('Worker: Loading scripts...');
        importScripts('/tf.min.js');
        importScripts('/coco-ssd.min.js');
        console.log('Worker: Scripts loaded');
        console.log('Worker: Loading model...');
        const model = await cocoSsd.load({ modelUrl: '/models/ssdlite_mobilenet_v2/model.json' });
        console.log('Worker: Model loaded successfully');

        console.log('Worker: Creating OffscreenCanvas...');
        const offscreenCanvas = new OffscreenCanvas(e.data.imageBitmap.width, e.data.imageBitmap.height);
        const ctx = offscreenCanvas.getContext('2d');
        if (!ctx) throw new Error('Failed to get OffscreenCanvas context');

        console.log('Worker: Drawing ImageBitmap...');
        ctx.drawImage(e.data.imageBitmap, 0, 0);
        const imageData = ctx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);

        console.log('Worker: Detecting objects...');
        const predictions = await model.detect(imageData);
        console.log('Worker: Predictions:', predictions);
        self.postMessage({ predictions });
    } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        console.error('Worker error:', errorMessage);
        self.postMessage({ error: `Worker error: ${errorMessage}` });
    }
};