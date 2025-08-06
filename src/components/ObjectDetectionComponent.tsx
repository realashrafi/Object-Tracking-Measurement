import React, { useRef, useState, useEffect, useCallback } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import { motion, AnimatePresence } from 'framer-motion';
import { Chart } from 'chart.js/auto';

interface DetectedObject {
    class: string;
    score: number;
    bbox: [number, number, number, number]; // [x, y, width, height]
}

interface ObjectDimensions {
    width: number;
    height: number;
    depth: number;
}

interface SizeOption {
    label: string;
    width: number;
    height: number;
    depth: number;
}

const ObjectTrackingComponent: React.FC = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const chartRef = useRef<HTMLCanvasElement>(null);
    const chartInstanceRef = useRef<Chart | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
    const [isDetecting, setIsDetecting] = useState<boolean>(false);
    const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [isCollecting, setIsCollecting] = useState<boolean>(false);
    const [collectedData, setCollectedData] = useState<DetectedObject[]>([]);
    const [dimensions, setDimensions] = useState<ObjectDimensions | null>(null);
    const [cameraDistanceCm, setCameraDistanceCm] = useState<number>(20);
    const [isBackCamera, setIsBackCamera] = useState<boolean>(true);
    const [useCamera, setUseCamera] = useState<boolean>(true);
    const [uploadedImage, setUploadedImage] = useState<HTMLImageElement | null>(null);
    const [step, setStep] = useState<number>(1);
    const [resolution, setResolution] = useState<{ width: number; height: number }>({ width: 1280, height: 720 });
    const [detectionInterval, setDetectionInterval] = useState<number>(300);
    const detectionIntervalRef = useRef<number | null>(null);
    const workerRef = useRef<Worker | null>(null);
    const [selectedSize, setSelectedSize] = useState<string>('');

    const resolutions = [
        { label: 'Auto', width: 0, height: 0 },
        { label: '480p', width: 640, height: 480 },
        { label: '720p', width: 1280, height: 720 },
        { label: '1080p', width: 1920, height: 1080 },
    ];

    const sizeOptions: SizeOption[] = [
        { label: 'Size 1', width: 15, height: 10, depth: 10 },
        { label: 'Size 2', width: 20, height: 15, depth: 10 },
        { label: 'Size 3', width: 20, height: 20, depth: 15 },
        { label: 'Size 4', width: 30, height: 20, depth: 20 },
        { label: 'Size 5', width: 35, height: 25, depth: 20 },
        { label: 'Size 6', width: 45, height: 25, depth: 20 },
        { label: 'Size 7', width: 40, height: 30, depth: 25 },
        { label: 'Size 8', width: 45, height: 40, depth: 30 },
        { label: 'Size 9', width: 55, height: 45, depth: 35 },
    ];

    // 获取支持的分辨率
    const getSupportedResolutions = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            const track = stream.getVideoTracks()[0];
            const capabilities = await track.getCapabilities();
            stream.getTracks().forEach((track) => track.stop());
            return {
                width: capabilities.width?.max || 1280,
                height: capabilities.height?.max || 720,
            };
        } catch (err) {
            console.error('Error getting supported resolutions:', err);
            return { width: 640, height: 480 }; // 回退
        }
    };

    // 在步骤2自动设置分辨率为Auto
    useEffect(() => {
        if (step === 2) {
            console.log('Setting resolution to Auto on step 2');
            setResolution({ width: 0, height: 0 });
        }
    }, [step]);

    // 加载COCO-SSD模型
    useEffect(() => {
        const loadModel = async () => {
            try {
                console.log('Loading TensorFlow backend...');
                await tf.setBackend('webgpu').catch(() => tf.setBackend('webgl'));
                console.log(`Using ${tf.getBackend()} backend`);
                console.log('Loading COCO-SSD model from local path...');
                const loadedModel = await cocoSsd.load({ modelUrl: '/models/ssdlite_mobilenet_v2/model.json' });
                setModel(loadedModel);
                setError(null);
                console.log('COCO-SSD model loaded successfully from local path');
            } catch (err: unknown) {
                const errorMessage = err instanceof Error ? err.message : 'Unknown error';
                setError(`Failed to load COCO-SSD model: ${errorMessage}. Check model files in /models/ssdlite_mobilenet_v2.`);
                console.error('Model load error:', err);
            }
        };
        loadModel();
    }, []);

    // 激活摄像头
    useEffect(() => {
        if (!useCamera) return;

        let retryCount = 0;
        const maxRetries = 3;

        const startCamera = async () => {
            try {
                console.log('Starting camera with resolution:', resolution);
                let constraints: MediaStreamConstraints = {
                    video: {
                        facingMode: isBackCamera ? 'environment' : 'user',
                    },
                };

                if (resolution.width === 0 && resolution.height === 0) {
                    const supportedRes = await getSupportedResolutions();
                    console.log('Supported resolution detected:', supportedRes);
                    constraints.video = {
                        // @ts-ignore
                        ...constraints.video,
                        width: { ideal: supportedRes.width },
                        height: { ideal: supportedRes.height },
                    };
                } else {
                    constraints.video = {
                        // @ts-ignore
                        ...constraints.video,
                        width: { ideal: resolution.width },
                        height: { ideal: resolution.height },
                    };
                }

                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.onloadedmetadata = () => {
                        console.log('Video metadata loaded, playing...');
                        videoRef.current?.play().catch((err) => {
                            setError(`Failed to play video: ${err.message}`);
                            console.error('Video play error:', err);
                        });
                    };
                    setError(null);
                } else {
                    setError('Video element not found.');
                    console.error('Video element not found');
                }
            } catch (err: unknown) {
                const errorMessage = err instanceof Error ? err.message : 'Unknown error';
                console.error('Camera error:', err);
                if (retryCount < maxRetries) {
                    retryCount++;
                    console.log(`Retrying camera initialization (${retryCount}/${maxRetries})...`);
                    setTimeout(startCamera, 1000);
                } else {
                    setError(`Unable to access camera after ${maxRetries} attempts: ${errorMessage}. Ensure camera permissions are granted.`);
                }
            }
        };
        startCamera();

        return () => {
            console.log('Cleaning up camera...');
            if (videoRef.current?.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach((track) => track.stop());
            }
        };
    }, [isBackCamera, resolution, useCamera]);

    // 设置Web Worker
    useEffect(() => {
        console.log('Initializing Web Worker...');
        workerRef.current = new Worker(new URL('./detectionWorker.js', import.meta.url));
        workerRef.current.onmessage = (e) => {
            console.log('Received message from worker:', e.data);
            if (e.data.error) {
                setError(e.data.error);
                console.error('Worker error:', e.data.error);
            } else {
                const filteredPredictions = e.data.predictions.filter((p: DetectedObject) => p.score > 0.5);
                console.log('Filtered predictions:', filteredPredictions);
                setDetectedObjects(filteredPredictions);
                if (isCollecting && filteredPredictions.length > 0) {
                    setCollectedData((prev) => [...prev, ...filteredPredictions]);
                    console.log('Collected data updated:', collectedData.length + filteredPredictions.length);
                }
            }
        };
        return () => {
            console.log('Terminating Web Worker...');
            workerRef.current?.terminate();
        };
    }, [isCollecting]);

    // 渲染检测到的物体（摄像头模式）
    const renderDetections = (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            setError('Failed to get canvas context.');
            console.error('Canvas context error');
            return;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;
        const scale = Math.min(scaleX, scaleY);
        const scaledWidth = video.videoWidth * scale;
        const scaledHeight = video.videoHeight * scale;
        const offsetX = (canvas.width - scaledWidth) / 2;
        const offsetY = (canvas.height - scaledHeight) / 2;

        ctx.drawImage(video, offsetX, offsetY, scaledWidth, scaledHeight);
        console.log('Raw video drawn on canvas with scaling:', { scale, offsetX, offsetY });

        detectedObjects.forEach((prediction) => {
            const [x, y, width, height] = prediction.bbox;
            const scaledX = x * scale + offsetX;
            const scaledY = y * scale + offsetY;
            const scaledWidth = width * scale;
            const scaledHeight = height * scale;

            ctx.strokeStyle = 'green';
            ctx.lineWidth = 2;
            ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
            ctx.fillStyle = 'green';
            ctx.font = '16px Arial';
            ctx.fillText(
                `${prediction.class} (${(prediction.score * 100).toFixed(1)}%)`,
                scaledX,
                scaledY > 10 ? scaledY - 5 : 10
            );
            console.log('Rendered detected object:', { ...prediction, scaledX, scaledY, scaledWidth, scaledHeight });
        });
    };

    // 渲染上传的图片并居中
    const trackAndCenterObject = (source: HTMLImageElement, canvas: HTMLCanvasElement) => {
        console.log('Rendering uploaded image on canvas with bbox:', detectedObjects[0]?.bbox);
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            setError('Failed to get canvas context.');
            console.error('Canvas context error');
            return;
        }

        const bbox = detectedObjects[0]?.bbox;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!bbox) {
            const scaleX = canvas.width / source.width;
            const scaleY = canvas.height / source.height;
            const scale = Math.min(scaleX, scaleY);
            const scaledWidth = source.width * scale;
            const scaledHeight = source.height * scale;
            const offsetX = (canvas.width - scaledWidth) / 2;
            const offsetY = (canvas.height - scaledHeight) / 2;

            ctx.drawImage(source, offsetX, offsetY, scaledWidth, scaledHeight);
            console.log('No detected object, drawing raw image with scaling:', { scale, offsetX, offsetY });
            return;
        }

        const [x, y, width, height] = bbox;
        const scaleX = canvas.width / width;
        const scaleY = canvas.height / height;
        const scale = Math.min(scaleX, scaleY, 1);
        const scaledWidth = source.width * scale;
        const scaledHeight = source.height * scale;

        const bboxCenterX = x + width / 2;
        const bboxCenterY = y + height / 2;
        const offsetX = (canvas.width - scaledWidth) / 2 - bboxCenterX * scale + width * scale / 2;
        const offsetY = (canvas.height - scaledHeight) / 2 - bboxCenterY * scale + height * scale / 2;

        ctx.drawImage(source, offsetX, offsetY, scaledWidth, scaledHeight);

        const scaledBboxX = (canvas.width - width * scale) / 2;
        const scaledBboxY = (canvas.height - height * scale) / 2;
        ctx.strokeStyle = 'green';
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledBboxX, scaledBboxY, width * scale, height * scale);

        ctx.fillStyle = 'green';
        ctx.font = '16px Arial';
        ctx.fillText(
            `${detectedObjects[0].class} (${(detectedObjects[0].score * 100).toFixed(1)}%)`,
            scaledBboxX,
            scaledBboxY > 10 ? scaledBboxY - 5 : 10
        );
        console.log('Rendered centered object:', { ...detectedObjects[0], scaledBboxX, scaledBboxY, scale });
    };

    const detectObjects = useCallback(() => {
        if (!model || !canvasRef.current || !isDetecting || !workerRef.current) {
            console.log('Detection skipped:', {
                model: !!model,
                canvas: !!canvasRef.current,
                isDetecting,
                worker: !!workerRef.current,
                video: !!videoRef.current,
                uploadedImage: !!uploadedImage,
                useCamera,
            });
            return;
        }

        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) {
            setError('Failed to get canvas context.');
            console.error('Canvas context error');
            return;
        }

        const source = useCamera && videoRef.current ? videoRef.current : uploadedImage;
        if (!source) {
            setError('No video or image source available.');
            console.error('No source available for detection');
            return;
        }

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
        tempCanvas.height = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) {
            setError('Failed to create temporary canvas context.');
            console.error('Temp canvas context error');
            return;
        }

        tempCtx.drawImage(source, 0, 0, tempCanvas.width, tempCanvas.height);

        createImageBitmap(tempCanvas).then((imageBitmap) => {
            console.log('ImageBitmap created, sending to worker...');
            workerRef.current!.postMessage({ imageBitmap }, [imageBitmap]);
            if (useCamera && videoRef.current) {
                renderDetections(videoRef.current, canvasRef.current!);
            } else if (!useCamera && uploadedImage) {
                trackAndCenterObject(uploadedImage, canvasRef.current!);
            }
        }).catch((err) => {
            setError(`Failed to create ImageBitmap: ${err.message}`);
            console.error('ImageBitmap error:', err);
        });
    }, [model, isDetecting, detectedObjects, useCamera, uploadedImage]);

    useEffect(() => {
        if (!isDetecting || (useCamera && !videoRef.current)) {
            console.log('Detection interval skipped:', { isDetecting, hasVideo: !!videoRef.current });
            return;
        }
        console.log('Starting detection interval...');
        detectionIntervalRef.current = window.setInterval(detectObjects, detectionInterval);
        return () => {
            console.log('Clearing detection interval...');
            if (detectionIntervalRef.current) {
                clearInterval(detectionIntervalRef.current);
                detectionIntervalRef.current = null;
            }
        };
    }, [isDetecting, detectObjects, detectionInterval, useCamera]);

    useEffect(() => {
        if (chartRef.current && collectedData.length > 0) {
            const ctx = chartRef.current.getContext('2d');
            if (ctx) {
                if (chartInstanceRef.current) {
                    chartInstanceRef.current.destroy();
                    chartInstanceRef.current = null;
                }

                chartInstanceRef.current = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: collectedData.map((_, i) => `Angle ${i + 1}`),
                        datasets: [
                            {
                                label: 'Width (px)',
                                data: collectedData.map((d) => d.bbox[2]),
                                borderColor: '#4CAF50',
                                fill: false,
                            },
                            {
                                label: 'Height (px)',
                                data: collectedData.map((d) => d.bbox[3]),
                                borderColor: '#2196F3',
                                fill: false,
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        scales: { y: { beginAtZero: true } },
                    },
                });
            }
        } else if (chartInstanceRef.current) {
            chartInstanceRef.current.destroy();
            chartInstanceRef.current = null;
        }

        return () => {
            if (chartInstanceRef.current) {
                chartInstanceRef.current.destroy();
                chartInstanceRef.current = null;
            }
        };
    }, [collectedData]);

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            console.log('Uploading image...');
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = () => {
                console.log('Image loaded successfully');
                setUploadedImage(img);
                setUseCamera(false);
                setError(null);
            };
            img.onerror = () => {
                setError('Failed to load uploaded image.');
                console.error('Image upload error');
            };
        }
    };

    const toggleCollection = () => {
        if (isCollecting) {
            setIsCollecting(false);
            calculateDimensions();
            setStep(3);
        } else {
            setCollectedData([]);
            setIsCollecting(true);
        }
    };

    const calculateDimensions = () => {
        if (collectedData.length < 3) {
            setError('Please collect data from at least 3 different angles.');
            console.error('Insufficient data: collectedData length is', collectedData.length);
            return;
        }

        const widths = collectedData.map((d) => d.bbox[2]);
        const heights = collectedData.map((d) => d.bbox[3]);
        const maxWidth = Math.max(...widths);
        const maxHeight = Math.max(...heights);
        const pixelToCmRatio = cameraDistanceCm / 500;
        const realWidth = maxWidth * pixelToCmRatio;
        const realHeight = maxHeight * pixelToCmRatio;
        const depthVariation = (Math.max(...widths) - Math.min(...widths)) * pixelToCmRatio;

        console.log('Calculated dimensions:', {
            maxWidth,
            maxHeight,
            pixelToCmRatio,
            realWidth,
            realHeight,
            depth: parseFloat(depthVariation.toFixed(2)),
        });

        setDimensions({
            width: parseFloat(realWidth.toFixed(2)),
            height: parseFloat(realHeight.toFixed(2)),
            depth: parseFloat(depthVariation.toFixed(2)),
        });

        // 推荐最接近的尺寸，仅基于width和height，优先 سایز بزرگ‌تر
        if (realWidth && realHeight) {
            const errorMargin = 0.05; // ضریب خطا 5%
            const closestSize = sizeOptions.find((size) => {
                const widthThreshold = size.width * (1 - errorMargin);
                const heightThreshold = size.height * (1 - errorMargin);
                return realWidth <= size.width && realHeight <= size.height && realWidth >= widthThreshold && realHeight >= heightThreshold;
            }) || sizeOptions.reduce((prev, curr) => {
                // اگر سایزی دقیقاً پیدا نشد، سایز بزرگ‌تر را انتخاب کن
                const prevFits = realWidth <= prev.width && realHeight <= prev.height;
                const currFits = realWidth <= curr.width && realHeight <= curr.height;
                if (currFits && !prevFits) return curr;
                if (prevFits && !currFits) return prev;
                const prevArea = prev.width * prev.height;
                const currArea = curr.width * curr.height;
                return currFits && prevFits && currArea < prevArea ? curr : prev;
            }, sizeOptions[sizeOptions.length - 1]);

            setSelectedSize(closestSize.label);
            console.log('Recommended size:', closestSize);
        }

        setError(null);
    };

    const calibrateDistance = () => {
        if (!detectedObjects[0]) {
            setError('No object detected for calibration.');
            console.error('Calibration failed: no detected object');
            return;
        }

        const referenceObjectWidthPx = detectedObjects[0].bbox[2];
        const referenceObjectRealWidthCm = 2.5;
        if (referenceObjectWidthPx < 10) {
            setError('Detected object is too small for reliable calibration.');
            console.error('Calibration failed: referenceObjectWidthPx too small', referenceObjectWidthPx);
            return;
        }

        const estimatedDistance = (referenceObjectRealWidthCm * 500) / referenceObjectWidthPx;
        const calibratedDistance = parseFloat(estimatedDistance.toFixed(2)) * 10 + 6;

        console.log('Calibration result:', {
            referenceObjectWidthPx,
            referenceObjectRealWidthCm,
            estimatedDistance,
            calibratedDistance,
        });

        setCameraDistanceCm(calibratedDistance);
        setError(null);
    };

    const exportData = () => {
        const data = JSON.stringify({ dimensions, collectedData, selectedSize });
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'object_data.json';
        link.click();
        URL.revokeObjectURL(url);
    };

    const resetDetection = () => {
        setIsDetecting(false);
        setIsCollecting(false);
        setDetectedObjects([]);
        setDimensions(null);
        setCollectedData([]);
        setUploadedImage(null);
        setUseCamera(true);
        setStep(1);
        setResolution({ width: 640, height: 480 });
        setSelectedSize('');
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
        if (chartInstanceRef.current) {
            chartInstanceRef.current.destroy();
            chartInstanceRef.current = null;
        }
    };

    const handleSizeSelection = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const selected = e.target.value;
        setSelectedSize(selected);
        const sizeData = sizeOptions.find((size) => size.label === selected);
        console.log('Selected size:', sizeData);
    };

    const renderStep = () => {
        switch (step) {
            case 1:
                return (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5 }}
                        className="p-6 bg-blue-50 rounded-lg text-center"
                    >
                        <h2 className="text-xl font-semibold mb-4">Welcome to Object Tracking</h2>
                        <p className="mb-4 text-gray-700">
                            Use your camera or upload an image to track objects and measure their dimensions. Rotate
                            around the object to capture multiple angles for accurate measurements.
                        </p>
                        <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => setStep(2)}
                            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium"
                        >
                            Start Tracking
                        </motion.button>
                    </motion.div>
                );
            case 2:
                return (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                        className="space-y-4"
                    >
                        <div className="relative w-full max-w-md aspect-video">
                            {useCamera ? (
                                <>
                                    <video
                                        ref={videoRef}
                                        className="w-full h-full rounded-lg object-contain"
                                        autoPlay
                                        playsInline
                                        style={{ transform: isBackCamera ? 'scaleX(1)' : 'scaleX(-1)' }}
                                    />
                                    <canvas
                                        ref={canvasRef}
                                        className="absolute top-0 left-0 w-full h-full"
                                        style={{ pointerEvents: 'none' }}
                                    />
                                </>
                            ) : (
                                <canvas
                                    ref={canvasRef}
                                    className="w-full h-full rounded-lg object-contain"
                                    style={{ pointerEvents: 'none' }}
                                />
                            )}
                        </div>
                        <div className="flex gap-2">
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => setUseCamera(true)}
                                className={`flex-1 px-4 py-2 rounded-lg text-white font-medium ${useCamera ? 'bg-blue-600' : 'bg-gray-400'}`}
                            >
                                Use Camera
                            </motion.button>
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => fileInputRef.current?.click()}
                                className={`flex-1 px-4 py-2 rounded-lg text-white font-medium ${!useCamera ? 'bg-blue-600' : 'bg-gray-400'}`}
                            >
                                Upload Image
                            </motion.button>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                onChange={handleImageUpload}
                                className="hidden"
                            />
                        </div>
                        <div className="grid grid-cols-1 gap-2">
                            <label className="text-sm font-medium text-gray-700">
                                Resolution:
                                <select
                                    value={resolution.width}
                                    onChange={(e) =>
                                        setResolution(resolutions.find((r) => r.width === parseInt(e.target.value)) || resolution)
                                    }
                                    className="mt-1 p-2 border rounded w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                >
                                    {resolutions.map((r) => (
                                        <option key={r.width} value={r.width}>
                                            {r.label}
                                        </option>
                                    ))}
                                </select>
                            </label>
                            <label className="text-sm font-medium text-gray-700">
                                Camera Distance (cm):
                                <input
                                    type="number"
                                    value={cameraDistanceCm}
                                    onChange={(e) => setCameraDistanceCm(parseFloat(e.target.value) || 50)}
                                    className="mt-1 p-2 border rounded w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    min="10"
                                    max="200"
                                />
                            </label>
                            <label className="text-sm font-medium text-gray-700">
                                Detection Interval (ms):
                                <input
                                    type="number"
                                    value={detectionInterval}
                                    onChange={(e) => setDetectionInterval(parseInt(e.target.value) || 300)}
                                    className="mt-1 p-2 border rounded w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    min="100"
                                    max="1000"
                                />
                            </label>
                        </div>
                        <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={calibrateDistance}
                            disabled={!detectedObjects.length}
                            className="w-full px-4 py-2 bg-teal-600 text-white rounded-lg font-medium disabled:bg-gray-400 disabled:cursor-not-allowed"
                        >
                            Auto-Calibrate Distance
                        </motion.button>
                        {isCollecting && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="p-2 bg-yellow-100 text-yellow-700 rounded text-sm text-center"
                            >
                                Collected angles: {collectedData.length}
                            </motion.div>
                        )}
                        <div className="flex gap-2">
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => {
                                    setIsDetecting(!isDetecting);
                                    console.log('Toggling detection:', !isDetecting);
                                }}
                                disabled={!model || (!useCamera && !uploadedImage)}
                                className={`flex-1 px-4 py-2 rounded-lg text-white font-medium ${
                                    !model || (!useCamera && !uploadedImage) ? 'bg-gray-400 cursor-not-allowed' : isDetecting ? 'bg-yellow-600' : 'bg-blue-600'
                                }`}
                            >
                                {isDetecting ? 'Stop Detection' : 'Start Detection'}
                            </motion.button>
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={toggleCollection}
                                disabled={!isDetecting || !detectedObjects.length}
                                className={`flex-1 px-4 py-2 rounded-lg text-white font-medium ${
                                    isDetecting && detectedObjects.length ? (isCollecting ? 'bg-orange-600' : 'bg-purple-600') : 'bg-gray-400 cursor-not-allowed'
                                }`}
                            >
                                {isCollecting ? 'Stop Collection' : 'Start Collection'}
                            </motion.button>
                            {useCamera && (
                                <motion.button
                                    whileHover={{ scale: 1.05 }}
                                    whileTap={{ scale: 0.95 }}
                                    onClick={() => setIsBackCamera(!isBackCamera)}
                                    className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg font-medium"
                                >
                                    Switch Camera
                                </motion.button>
                            )}
                        </div>
                        <canvas ref={chartRef} className="w-full" />
                    </motion.div>
                );
            case 3:
                return (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5 }}
                        className="p-6 bg-green-50 rounded-lg text-center"
                    >
                        <h2 className="text-xl font-semibold mb-4">Measurement Results</h2>
                        {dimensions && (
                            <div className="mb-4">
                                <p className="text-lg font-medium">
                                    Width: <span className="text-green-600">{dimensions.width} cm</span>
                                </p>
                                <p className="text-lg font-medium">
                                    Height: <span className="text-green-600">{dimensions.height} cm</span>
                                </p>
                                <p className="text-lg font-medium">
                                    Depth: <span className="text-green-600">{dimensions.depth} cm</span>
                                </p>
                            </div>
                        )}
                        <div className="mb-4">
                            <label className="text-sm font-medium text-gray-700">
                                Select Size:
                                <select
                                    value={selectedSize}
                                    onChange={handleSizeSelection}
                                    className="mt-1 p-2 border rounded w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                >
                                    {sizeOptions.map((size) => (
                                        <option key={size.label} value={size.label}>
                                            {size.label} ({size.width} × {size.height} × {size.depth} cm)
                                        </option>
                                    ))}
                                </select>
                            </label>
                        </div>
                        <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={exportData}
                            className="px-6 py-3 bg-green-600 text-white rounded-lg font-medium mb-4"
                        >
                            Export Data
                        </motion.button>
                        <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={resetDetection}
                            className="px-6 py-3 bg-red-600 text-white rounded-lg font-medium"
                        >
                            Start Over
                        </motion.button>
                    </motion.div>
                );
            default:
                return null;
        }
    };

    return (
        <div className="flex flex-col items-center p-4 bg-gray-100 min-h-screen">
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="bg-white p-6 rounded-lg shadow-lg w-full max-w-md"
            >
                <h1 className="text-2xl font-bold mb-4 text-center">Object Tracking & Measurement</h1>
                <AnimatePresence>
                    {error && (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="mb-4 p-2 bg-red-100 text-red-700 rounded text-sm"
                        >
                            {error}
                            {error.includes('model') && (
                                <p>Check if the model files are correctly placed in the /models/ssdlite_mobilenet_v2 directory.</p>
                            )}
                            {error.includes('camera') && (
                                <p>Ensure camera permissions are granted in your browser settings.</p>
                            )}
                            {!detectedObjects.length && isDetecting && (
                                <p>No object detected. Try a well-lit environment or a different image.</p>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
                {renderStep()}
            </motion.div>
        </div>
    );
};

export default ObjectTrackingComponent;