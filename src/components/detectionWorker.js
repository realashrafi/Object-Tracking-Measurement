//@ts-nocheck
/* eslint-disable */
self.onmessage = async (e) => {
    try {
        console.log('Worker: Loading scripts...');
        importScripts('/tf.min.js');
        importScripts('/coco-ssd.min.js');
        console.log('Worker: Scripts loaded');

        // ✅ Dynamic HTTPS URL
        const origin = self.location.origin || 'https://size.zboom.ir'; // fallback
        const modelUrl = `${origin}/models/ssdlite_mobilenet_v2/model.json`;

        console.log('Worker: Loading model from:', modelUrl);
        console.log('Worker: Current origin:', origin);

        const model = await cocoSsd.load({ modelUrl });
        console.log('Worker: Model loaded successfully');

        // بقیه کد...
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