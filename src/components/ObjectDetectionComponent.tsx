//@ts-nocheck
import React, {useRef, useState, useEffect, useCallback} from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import {motion, AnimatePresence} from 'framer-motion';
import {Chart} from 'chart.js/auto';
import Loading from "./Loading";
import {ReactComponent as Logo} from "../components/svg/zboomLogoSolo.svg";
import {FaPause, FaPlay, FaCamera, FaUpload,FaDownload, FaUndo} from "react-icons/fa";
import {RiCameraSwitchFill} from "react-icons/ri";
import {MdOutlineAutoMode, MdRotate90DegreesCw} from "react-icons/md";

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
    const [resolution, setResolution] = useState<{ width: number; height: number }>({width: 1280, height: 720});
    const [detectionInterval, setDetectionInterval] = useState<number>(300);
    const detectionIntervalRef = useRef<number | null>(null);
    const workerRef = useRef<Worker | null>(null);
    const [selectedSize, setSelectedSize] = useState<string>('');
    const [loading, setLoading] = useState(true);

    // اضافه کردن state برای returnUrl
    const [returnUrl, setReturnUrl] = useState<string | null>(null);
    const [selectedSizeOption, setSelectedSizeOption] = useState<SizeOption | null>(null);

    const resolutions = [
        {label: 'خودکار', width: 0, height: 0},
        {label: '480p', width: 640, height: 480},
        {label: '720p', width: 1280, height: 720},
        {label: '1080p', width: 1920, height: 1080},
    ];

    const sizeOptions: SizeOption[] = [
        {label: 'سایز ۱', width: 15, height: 10, depth: 10},
        {label: 'سایز ۲', width: 20, height: 15, depth: 10},
        {label: 'سایز ۳', width: 20, height: 20, depth: 15},
        {label: 'سایز ۴', width: 30, height: 20, depth: 20},
        {label: 'سایز ۵', width: 35, height: 25, depth: 20},
        {label: 'سایز ۶', width: 45, height: 25, depth: 20},
        {label: 'سایز ۷', width: 40, height: 30, depth: 25},
        {label: 'سایز ۸', width: 45, height: 40, depth: 30},
        {label: 'سایز ۹', width: 55, height: 45, depth: 35},
    ];

    // خواندن returnUrl از URL params
    useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const returnUrlParam = urlParams.get('returnUrl');

        if (returnUrlParam) {
            try {
                const decodedUrl = decodeURIComponent(returnUrlParam);
                setReturnUrl(decodedUrl);
                console.log('Return URL استخراج شد:', decodedUrl);
            } catch (err) {
                console.error('خطا در decode کردن returnUrl:', err);
                setError('خطا در خواندن آدرس بازگشت');
            }
        }
    }, []);

    useEffect(() => {
        const timeoutId = setTimeout(() => {
            setLoading(false)
        }, 2000);

        return () => clearTimeout(timeoutId);
    }, []);

    const getSupportedResolutions = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            const track = stream.getVideoTracks()[0];
            const capabilities = await track.getCapabilities();
            stream.getTracks().forEach((track) => track.stop());
            return {
                width: capabilities.width?.max || 1280,
                height: capabilities.height?.max || 720,
            };
        } catch (err) {
            console.error('خطا در دریافت وضوح‌های پشتیبانی‌شده:', err);
            return {width: 640, height: 480};
        }
    };

    useEffect(() => {
        if (step === 2) {
            console.log('تنظیم رزولوشن به خودکار در گام دوم');
            setResolution({width: 0, height: 0});
        }
    }, [step]);

    useEffect(() => {
        const loadModel = async () => {
            try {
                console.log('در حال بارگذاری بک‌اند TensorFlow...');
                await tf.setBackend('webgpu').catch(() => tf.setBackend('webgl'));
                console.log(`استفاده از بک‌اند ${tf.getBackend()}`);
                console.log('بارگذاری مدل COCO-SSD از مسیر محلی...');
                const loadedModel = await cocoSsd.load({modelUrl: '/models/ssdlite_mobilenet_v2/model.json'});
                setModel(loadedModel);
                setError(null);
                console.log('مدل COCO-SSD با موفقیت از مسیر محلی بارگذاری شد');
            } catch (err: unknown) {
                const errorMessage = err instanceof Error ? err.message : 'خطای ناشناخته';
                setError(`بارگذاری مدل تشخیص اشیاء با مشکل مواجه شد: ${errorMessage}. لطفاً بررسی کنید که فایل‌های مدل به‌درستی در پوشه /models/ssdlite_mobilenet_v2 قرار گرفته باشند.`);
                console.error('خطای بارگذاری مدل:', err);
            }
        };
        loadModel();
    }, []);

    useEffect(() => {
        if (!useCamera) return;

        let retryCount = 0;
        const maxRetries = 3;

        const startCamera = async () => {
            try {
                console.log('راه‌اندازی دوربین با رزولوشن:', resolution);
                let constraints: MediaStreamConstraints = {
                    video: {
                        facingMode: isBackCamera ? 'environment' : 'user',
                    },
                };

                if (resolution.width === 0 && resolution.height === 0) {
                    const supportedRes = await getSupportedResolutions();
                    console.log('رزولوشن پشتیبانی‌شده شناسایی شد:', supportedRes);
                    constraints.video = {
                        // @ts-ignore
                        ...constraints.video,
                        width: {ideal: supportedRes.width},
                        height: {ideal: supportedRes.height},
                    };
                } else {
                    constraints.video = {
                        // @ts-ignore
                        ...constraints.video,
                        width: {ideal: resolution.width},
                        height: {ideal: resolution.height},
                    };
                }

                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.onloadedmetadata = () => {
                        console.log('متادیتای ویدئو بارگذاری شد، در حال پخش...');
                        videoRef.current?.play().catch((err) => {
                            setError(`پخش ویدئو با مشکل مواجه شد: ${err.message}`);
                            console.error('خطای پخش ویدئو:', err);
                        });
                    };
                    setError(null);
                } else {
                    console.error('المان ویدئو یافت نشد');
                }
            } catch (err: unknown) {
                const errorMessage = err instanceof Error ? err.message : 'خطای ناشناخته';
                console.error('خطای دوربین:', err);
                if (retryCount < maxRetries) {
                    retryCount++;
                    console.log(`تلاش مجدد برای راه‌اندازی دوربین (${retryCount}/${maxRetries})...`);
                    setTimeout(startCamera, 1000);
                } else {
                    setError(`دسترسی به دوربین پس از ${maxRetries} تلاش ممکن نشد: ${errorMessage}. لطفاً مطمئن شوید که دسترسی به دوربین در تنظیمات مرورگر شما فعال است.`);
                }
            }
        };
        startCamera();

        return () => {
            console.log('پاک‌سازی دوربین...');
            if (videoRef.current?.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach((track) => track.stop());
            }
        };
    }, [isBackCamera, resolution, useCamera]);

    useEffect(() => {
        console.log('راه‌اندازی Web Worker...');
        workerRef.current = new Worker(new URL('./detectionWorker.js', import.meta.url));
        workerRef.current.onmessage = (e) => {
            console.log('پیام دریافت‌شده از worker:', e.data);
            if (e.data.error) {
                setError(e.data.error);
                console.error('خطای worker:', e.data.error);
            } else {
                const filteredPredictions = e.data.predictions.filter((p: DetectedObject) => p.score > 0.5);
                console.log('پیش‌بینی‌های فیلترشده:', filteredPredictions);
                setDetectedObjects(filteredPredictions);
                if (isCollecting && filteredPredictions.length > 0) {
                    setCollectedData((prev) => [...prev, ...filteredPredictions]);
                    console.log('داده‌های جمع‌آوری‌شده به‌روزرسانی شد:', collectedData.length + filteredPredictions.length);
                }
            }
        };
        return () => {
            console.log('خاتمه دادن به Web Worker...');
            workerRef.current?.terminate();
        };
    }, [isCollecting]);

    const renderDetections = (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            setError('دریافت محتوای بوم ممکن نشد.');
            console.error('خطای محتوای بوم');
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
        console.log('ویدئوی خام روی بوم با مقیاس‌بندی رسم شد:', {scale, offsetX, offsetY});

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
            ctx.font = '16px Vazirmatn, Arial';
            ctx.fillText(
                `${prediction.class} (${(prediction.score * 100).toFixed(1)}%)`,
                scaledX,
                scaledY > 10 ? scaledY - 5 : 10
            );
            console.log('شیء شناسایی‌شده رندر شد:', {...prediction, scaledX, scaledY, scaledWidth, scaledHeight});
        });
    };

    const trackAndCenterObject = (source: HTMLImageElement, canvas: HTMLCanvasElement) => {
        console.log('رندر تصویر آپلود شده روی بوم با bbox:', detectedObjects[0]?.bbox);
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            setError('دریافت محتوای بوم ممکن نشد.');
            console.error('خطای محتوای بوم');
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
            console.log('هیچ شیء شناسایی نشد، رسم تصویر خام با مقیاس‌بندی:', {scale, offsetX, offsetY});
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
        ctx.font = '16px Vazirmatn, Arial';
        ctx.fillText(
            `${detectedObjects[0].class} (${(detectedObjects[0].score * 100).toFixed(1)}%)`,
            scaledBboxX,
            scaledBboxY > 10 ? scaledBboxY - 5 : 10
        );
        console.log('شیء متمرکز رندر شد:', {...detectedObjects[0], scaledBboxX, scaledBboxY, scale});
    };

    const detectObjects = useCallback(() => {
        if (!model || !canvasRef.current || !isDetecting || !workerRef.current) {
            console.log('تشخیص متوقف شد:', {
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
            setError('دریافت محتوای بوم ممکن نشد.');
            console.error('خطای محتوای بوم');
            return;
        }

        const source = useCamera && videoRef.current ? videoRef.current : uploadedImage;
        if (!source) {
            setError('منبع ویدئو یا تصویر در دسترس نیست.');
            console.error('منبع برای تشخیص در دسترس نیست');
            return;
        }

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
        tempCanvas.height = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) {
            setError('ایجاد محتوای موقت بوم ممکن نشد.');
            console.error('خطای محتوای بوم موقت');
            return;
        }

        tempCtx.drawImage(source, 0, 0, tempCanvas.width, tempCanvas.height);

        createImageBitmap(tempCanvas).then((imageBitmap) => {
            console.log('ImageBitmap ایجاد شد، ارسال به worker...');
            workerRef.current!.postMessage({imageBitmap}, [imageBitmap]);
            if (useCamera && videoRef.current) {
                renderDetections(videoRef.current, canvasRef.current!);
            } else if (!useCamera && uploadedImage) {
                trackAndCenterObject(uploadedImage, canvasRef.current!);
            }
        }).catch((err) => {
            setError(`ایجاد ImageBitmap ممکن نشد: ${err.message}`);
            console.error('خطای ImageBitmap:', err);
        });
    }, [model, isDetecting, detectedObjects, useCamera, uploadedImage]);

    useEffect(() => {
        if (!isDetecting || (useCamera && !videoRef.current)) {
            console.log('بازه تشخیص متوقف شد:', {isDetecting, hasVideo: !!videoRef.current});
            return;
        }
        console.log('شروع بازه تشخیص...');
        detectionIntervalRef.current = window.setInterval(detectObjects, detectionInterval);
        return () => {
            console.log('پاک‌سازی بازه تشخیص...');
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
                        labels: collectedData.map((_, i) => `زاویه ${i + 1}`),
                        datasets: [
                            {
                                label: 'عرض (پیکسل)',
                                data: collectedData.map((d) => d.bbox[2]),
                                borderColor: '#FFC20E',
                                fill: false,
                            },
                            {
                                label: 'ارتفاع (پیکسل)',
                                data: collectedData.map((d) => d.bbox[3]),
                                borderColor: '#073054',
                                fill: false,
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        scales: {y: {beginAtZero: true}},
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
            console.log('در حال آپلود تصویر...');
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = () => {
                console.log('تصویر با موفقیت بارگذاری شد');
                setUploadedImage(img);
                setUseCamera(false);
                setError(null);
            };
            img.onerror = () => {
                setError('بارگذاری تصویر آپلود شده با مشکل مواجه شد.');
                console.error('خطای آپلود تصویر');
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
            setError('لطفاً داده‌ها را از حداقل ۳ زاویه مختلف جمع‌آوری کنید.');
            console.error('داده‌های ناکافی: طول collectedData برابر است با', collectedData.length);
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

        console.log('ابعاد محاسبه‌شده:', {
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

        if (realWidth && realHeight) {
            const errorMargin = 0.05;
            const closestSize = sizeOptions.find((size) => {
                const widthThreshold = size.width * (1 - errorMargin);
                const heightThreshold = size.height * (1 - errorMargin);
                return realWidth <= size.width && realHeight <= size.height && realWidth >= widthThreshold && realHeight >= heightThreshold;
            }) || sizeOptions.reduce((prev, curr) => {
                const prevFits = realWidth <= prev.width && realHeight <= prev.height;
                const currFits = realWidth <= curr.width && realHeight <= curr.height;
                if (currFits && !prevFits) return curr;
                if (prevFits && !currFits) return prev;
                const prevArea = prev.width * prev.height;
                const currArea = curr.width * curr.height;
                return currFits && prevFits && currArea < prevArea ? curr : prev;
            }, sizeOptions[sizeOptions.length - 1]);

            setSelectedSize(closestSize.label);
            setSelectedSizeOption(closestSize);
            console.log('سایز پیشنهادی:', closestSize);
        }

        setError(null);
    };

    const calibrateDistance = () => {
        if (!detectedObjects[0]) {
            setError('هیچ شیء برای کالیبراسیون شناسایی نشد.');
            console.error('کالیبراسیون ناموفق: هیچ شیء شناسایی نشد');
            return;
        }

        const referenceObjectWidthPx = detectedObjects[0].bbox[2];
        const referenceObjectRealWidthCm = 2.5;
        if (referenceObjectWidthPx < 10) {
            setError('شیء شناسایی‌شده برای کالیبراسیون قابل اعتماد خیلی کوچک است.');
            console.error('کالیبراسیون ناموفق: referenceObjectWidthPx خیلی کوچک است', referenceObjectWidthPx);
            return;
        }

        const estimatedDistance = (referenceObjectRealWidthCm * 500) / referenceObjectWidthPx;
        const calibratedDistance = parseFloat(estimatedDistance.toFixed(2)) * 10 + 6;

        console.log('نتیجه کالیبراسیون:', {
            referenceObjectWidthPx,
            referenceObjectRealWidthCm,
            estimatedDistance,
            calibratedDistance,
        });

        setCameraDistanceCm(calibratedDistance);
        setError(null);
    };

    // تابع exportData بهبود یافته
    const exportData = () => {
        // اگر returnUrl وجود داره، redirect کن
        if (returnUrl && dimensions && selectedSizeOption) {
            try {
                const redirectUrl = new URL(returnUrl);

                // تنظیم parameters اصلی
                redirectUrl.searchParams.set('length', selectedSizeOption.width.toString());
                redirectUrl.searchParams.set('width', selectedSizeOption.height.toString());
                redirectUrl.searchParams.set('height', selectedSizeOption.depth.toString());

                // اضافه کردن ابعاد محاسبه‌شده
                if (dimensions) {
                    redirectUrl.searchParams.set('measuredWidth', dimensions.width.toString());
                    redirectUrl.searchParams.set('measuredHeight', dimensions.height.toString());
                    redirectUrl.searchParams.set('measuredDepth', dimensions.depth.toString());
                }

                // اضافه کردن سایز انتخاب‌شده
                redirectUrl.searchParams.set('selectedSize', selectedSize);

                console.log('Redirect به:', redirectUrl.toString());
                console.log('Parameters:', Object.fromEntries(redirectUrl.searchParams));

                // redirect کردن
                window.location.href = redirectUrl.toString();
                return;
            } catch (err) {
                console.error('خطا در ساخت URL redirect:', err);
                setError('خطا در بازگشت به آدرس مقصد');
            }
        }

        // اگر returnUrl وجود نداره، مثل قبل export JSON کن
        const data = JSON.stringify({dimensions, collectedData, selectedSize});
        const blob = new Blob([data], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'object_data.json';
        link.click();
        URL.revokeObjectURL(url);

        console.log('داده‌ها به صورت JSON دانلود شد');
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
        setResolution({width: 640, height: 480});
        setSelectedSize('');
        setSelectedSizeOption(null);
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
        setSelectedSizeOption(sizeData);
        console.log('سایز انتخاب‌شده:', sizeData);
        console.log('Return URL موجود:', returnUrl);

        // Preview URL در console
        if (returnUrl && sizeData) {
            const previewUrl = new URL(returnUrl);
            previewUrl.searchParams.set('length', sizeData.width.toString());
            previewUrl.searchParams.set('width', sizeData.height.toString());
            previewUrl.searchParams.set('height', sizeData.depth.toString());
            console.log('Preview URL:', previewUrl.toString());
        }
    };

    const renderStep = () => {
        switch (step) {
            case 1:
                return (
                    <motion.div
                        initial={{opacity: 0, scale: 0.8}}
                        animate={{opacity: 1, scale: 1}}
                        transition={{duration: 0.5}}
                        className="p-6 bg-white rounded-lg text-center relative overflow-hidden"
                        style={{direction: 'rtl', fontFamily: 'Vazirmatn, sans-serif'}}
                    >
                        <h2 className="text-xl font-semibold mb-4">به ردیابی اشیاء خوش آمدید</h2>
                        <p className="mb-4 text-gray-700 z-10">
                            از دوربین خود یا یک تصویر آپلود شده برای ردیابی اشیاء و اندازه‌گیری ابعاد آنها استفاده کنید.
                            برای اندازه‌گیری دقیق، شیء را از زوایای مختلف اسکن کنید.
                        </p>
                        <Logo className="absolute w-30 opacity-50 -bottom-2 left-1 rotate-[25deg] h-30  "/>
                        <motion.button
                            whileHover={{scale: 1.05}}
                            whileTap={{scale: 0.95}}
                            onClick={() => setStep(2)}
                            className="px-6 py-3 bg-[#FFC20E] text-[#073054] rounded-lg font-medium"
                        >
                            شروع ردیابی
                        </motion.button>
                    </motion.div>
                );
            case 2:
                return (
                    <motion.div
                        initial={{opacity: 0, y: 20}}
                        animate={{opacity: 1, y: 0}}
                        transition={{duration: 0.5}}
                        className="space-y-4"
                        style={{direction: 'rtl', fontFamily: 'Vazirmatn, sans-serif'}}
                    >
                        <div className="relative w-full max-w-md aspect-video">
                            {useCamera ? (
                                <>
                                    <video
                                        ref={videoRef}
                                        className="w-full h-full rounded-lg object-contain"
                                        autoPlay
                                        playsInline
                                        style={{transform: isBackCamera ? 'scaleX(1)' : 'scaleX(-1)'}}
                                    />
                                    <canvas
                                        ref={canvasRef}
                                        className="absolute top-0 left-0 w-full h-full"
                                        style={{pointerEvents: 'none'}}
                                    />
                                </>
                            ) : (
                                <canvas
                                    ref={canvasRef}
                                    className="w-full h-full rounded-lg object-contain"
                                    style={{pointerEvents: 'none'}}
                                />
                            )}
                        </div>
                        <div className="flex items-center justify-center gap-2">
                            <motion.button
                                whileHover={{scale: 1.05}}
                                whileTap={{scale: 0.95}}
                                onClick={() => setUseCamera(true)}
                                className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-[#073054] font-medium ${useCamera ? 'bg-[#FFC20E]' : 'bg-gray-400'}`}
                            >
                                <FaCamera/>استفاده از دوربین
                            </motion.button>
                            <motion.button
                                whileHover={{scale: 1.05}}
                                whileTap={{scale: 0.95}}
                                onClick={() => fileInputRef.current?.click()}
                                className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-[#073054] font-medium ${!useCamera ? 'bg-[#FFC20E]' : 'bg-gray-400'}`}
                            >
                                <FaUpload/>آپلود تصویر
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
                                وضوح تصویر:
                                <select
                                    value={resolution.width}
                                    onChange={(e) =>
                                        setResolution(resolutions.find((r) => r.width === parseInt(e.target.value)) || resolution)
                                    }
                                    className="mt-1 p-2 bg-white rounded w-full text-sm focus:outline-none focus:ring-2 focus:ring-[#073054]"
                                >
                                    {resolutions.map((r) => (
                                        <option key={r.width} value={r.width}>
                                            {r.label}
                                        </option>
                                    ))}
                                </select>
                            </label>
                            <label className="text-sm font-medium text-gray-700">
                                فاصله دوربین (سانتی‌متر):
                                <input
                                    type="number"
                                    value={cameraDistanceCm}
                                    onChange={(e) => setCameraDistanceCm(parseFloat(e.target.value) || 50)}
                                    className="mt-1 p-2 bg-white rounded w-full text-sm focus:outline-none focus:ring-2 focus:ring-[#073054]"
                                    min="10"
                                    max="200"
                                />
                            </label>
                            <label className="text-sm font-medium text-gray-700">
                                بازه تشخیص (میلی‌ثانیه):
                                <input
                                    type="number"
                                    value={detectionInterval}
                                    onChange={(e) => setDetectionInterval(parseInt(e.target.value) || 300)}
                                    className="mt-1 p-2 bg-white rounded w-full text-sm focus:outline-none focus:ring-2 focus:ring-[#073054]"
                                    min="100"
                                    max="1000"
                                />
                            </label>
                        </div>
                        <motion.button
                            whileHover={{scale: 1.05}}
                            whileTap={{scale: 0.95}}
                            onClick={calibrateDistance}
                            disabled={!detectedObjects.length}
                            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-[#FFC20E] text-[#073054] rounded-lg font-medium disabled:bg-gray-400 disabled:cursor-not-allowed"
                        >
                            <MdOutlineAutoMode/> کالیبراسیون خودکار فاصله
                        </motion.button>
                        {isCollecting && (
                            <motion.div
                                initial={{opacity: 0}}
                                animate={{opacity: 1}}
                                className="flex items-center justify-center gap-2 p-2 bg-[#FFC20E] text-[#073054] rounded text-sm text-center"
                            >
                                <MdRotate90DegreesCw/>زوایای جمع‌آوری‌شده: {collectedData.length}
                            </motion.div>
                        )}
                        <div className="flex gap-2">
                            <motion.button
                                whileHover={{scale: 1.05}}
                                whileTap={{scale: 0.95}}
                                onClick={() => {
                                    setIsDetecting(!isDetecting);
                                    console.log('تغییر وضعیت تشخیص:', !isDetecting);
                                }}
                                disabled={!model || (!useCamera && !uploadedImage)}
                                className={`flex-1 px-4 py-2 rounded-lg text-[#073054] font-medium ${
                                    !model || (!useCamera && !uploadedImage) ? 'bg-gray-400 cursor-not-allowed' : isDetecting ? 'bg-[#FFC20E]' : 'bg-[#FFC20E]'
                                }`}
                            >
                                {isDetecting ?
                                    <div className={'flex items-center justify-center gap-2'}><FaPause/>توقف تشخیص
                                    </div> :
                                    <div className={'flex items-center justify-center gap-2'}><FaPlay/>
                                        شروع تشخیص </div>}
                            </motion.button>
                            <motion.button
                                whileHover={{scale: 1.05}}
                                whileTap={{scale: 0.95}}
                                onClick={toggleCollection}
                                disabled={!isDetecting || !detectedObjects.length}
                                className={`flex-1 px-4 py-2 rounded-lg text-[#073054] font-medium ${
                                    isDetecting && detectedObjects.length ? (isCollecting ? 'bg-[#FFC20E]' : 'bg-[#FFC20E]') : 'bg-gray-400 cursor-not-allowed'
                                }`}
                            >
                                {isCollecting ?
                                    <div className={'flex items-center justify-center gap-2'}><FaPause/>توقف جمع‌آوری
                                    </div> :
                                    <div className={'flex items-center justify-center gap-2'}><FaPlay/>
                                        شروع جمع‌آوری</div>}
                            </motion.button>
                            {useCamera && (
                                <motion.button
                                    whileHover={{scale: 1.05}}
                                    whileTap={{scale: 0.95}}
                                    onClick={() => setIsBackCamera(!isBackCamera)}
                                    className="flex items-center justify-center gap-2 px-4 py-2 bg-[#FFC20E] text-[#073054] rounded-lg font-medium"
                                >
                                    <RiCameraSwitchFill className={'w-5 h-5'}/>تغییر دوربین
                                </motion.button>
                            )}
                        </div>
                        <canvas ref={chartRef} className="w-full"/>
                    </motion.div>
                );
            case 3:
                return (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5 }}
                        className="p-6 bg-white rounded-xl text-center relative"
                        style={{ direction: 'rtl', fontFamily: 'Vazirmatn, sans-serif' }}
                    >
                        <h2 className="text-2xl font-bold mb-6 text-[#073054]">نتایج اندازه‌گیری</h2>

                        {/* نمایش returnUrl برای debug */}
                        {returnUrl && (
                            <motion.div
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="mb-4 p-3 bg-blue-50 border border-blue-200 text-blue-700 rounded-lg text-sm"
                            >
                                <div className="font-medium mb-1">آدرس بازگشت:</div>
                                <div className="text-xs break-all opacity-80">{returnUrl}</div>
                            </motion.div>
                        )}

                        {dimensions && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="mb-6 p-4 bg-gray-50 rounded-lg relative overflow-hidden border border-gray-200"
                            >
                                <Logo className="absolute w-30 opacity-50 -bottom-2 left-1 rotate-[15deg] h-30  "/>
                                <div className="grid grid-cols-3 gap-2 text-center">
                                    <div>
                                        <p className="text-sm text-gray-600">عرض</p>
                                        <p className="text-lg font-semibold text-[#073054]">{dimensions.width} cm</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-600">ارتفاع</p>
                                        <p className="text-lg font-semibold text-[#073054]">{dimensions.height} cm</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-600">عمق</p>
                                        <p className="text-lg font-semibold text-[#073054]">{dimensions.depth} cm</p>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        <div className="mb-6">
                            <label className="text-sm font-medium text-gray-700 block mb-2">
                                انتخاب سایز:
                                <select
                                    value={selectedSize}
                                    onChange={handleSizeSelection}
                                    className="mt-1 p-3 border rounded-lg w-full text-sm focus:outline-none focus:ring-2 focus:ring-[#073054] bg-white text-[#073054]"
                                    disabled={!dimensions}
                                >
                                    {!dimensions && (
                                        <option value="">لطفاً ابتدا اندازه‌گیری کنید</option>
                                    )}
                                    {sizeOptions.map((size) => (
                                        <option key={size.label} value={size.label} className="text-[#073054]">
                                            {size.label} ({size.width} × {size.height} × {size.depth} cm)
                                        </option>
                                    ))}
                                </select>
                            </label>
                        </div>

                        {/* Preview URL برای debug */}
                        {returnUrl && dimensions && selectedSizeOption && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="mb-4 p-3 bg-green-50 border border-green-200 text-green-700 rounded-lg text-sm"
                            >
                                <div className="font-medium mb-1">پیش‌نمایش آدرس مقصد:</div>
                                <div className="text-xs break-all">
                                    <code className="bg-green-100 px-1 py-0.5 rounded">
                                        {returnUrl}
                                        ?length={selectedSizeOption.width}
                                        &width={selectedSizeOption.height}
                                        &height={selectedSizeOption.depth}
                                        {dimensions && (
                                            <>
                                                &measuredWidth={dimensions.width}
                                                &measuredHeight={dimensions.height}
                                                &measuredDepth={dimensions.depth}
                                            </>
                                        )}
                                        &selectedSize={encodeURIComponent(selectedSize)}
                                    </code>
                                </div>
                            </motion.div>
                        )}

                        <div className={'flex items-center justify-center gap-2 flex-col sm:flex-row'}>
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={exportData}
                                disabled={!dimensions || !selectedSize}
                                className={`w-full sm:w-auto px-6 py-3 rounded-lg font-medium flex items-center justify-center gap-2 ${
                                    dimensions && selectedSize
                                        ? 'bg-[#FFC20E] text-[#073054] hover:bg-[#e6b100]'
                                        : 'bg-gray-400 text-gray-500 cursor-not-allowed'
                                }`}
                            >
                                <FaDownload />
                                {returnUrl ? 'بازگشت به بسته‌بندی' : 'صادر کردن داده‌ها'}
                            </motion.button>

                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={resetDetection}
                                className="w-full sm:w-auto px-6 py-3 bg-[#073054] text-white rounded-lg font-medium flex items-center justify-center gap-2 hover:bg-[#052548]"
                            >
                                <FaUndo /> شروع مجدد
                            </motion.button>
                        </div>

                        {/* نمایش وضعیت redirect */}
                        {returnUrl && dimensions && selectedSize && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="mt-4 p-2 bg-blue-50 text-blue-700 rounded text-xs text-center"
                            >
                                ✅ بعد از کلیک روی دکمه، به آدرس بسته‌بندی با ابعاد محاسبه‌شده برمی‌گردید
                            </motion.div>
                        )}
                    </motion.div>
                );
            default:
                return null;
        }
    };

    return (
        <div className="flex justify-center flex-col items-center p-4 bg-white min-h-screen"
             style={{direction: 'rtl', fontFamily: 'Vazirmatn, sans-serif'}}>
            {loading && <Loading/>}
            <motion.div
                initial={{opacity: 0, y: -20}}
                animate={{opacity: 1, y: 0}}
                transition={{duration: 0.5}}
                className="bg-gray-100 p-6 rounded-lg shadow-lg w-full max-w-md"
            >
                <h1 className="text-2xl font-bold mb-4 text-center">ردیابی و اندازه‌گیری اشیاء</h1>
                <AnimatePresence>
                    {error && (
                        <motion.div
                            initial={{opacity: 0, y: -10}}
                            animate={{opacity: 1, y: 0}}
                            exit={{opacity: 0, y: -10}}
                            className="mb-4 p-2 bg-red-100 text-red-700 rounded text-sm"
                        >
                            {error}
                            {error.includes('مدل') && (
                                <p>لطفاً بررسی کنید که فایل‌های مدل به‌درستی در پوشه /models/ssdlite_mobilenet_v2 قرار
                                    گرفته باشند.</p>
                            )}
                            {error.includes('دوربین') && (
                                <p>مطمئن شوید که دسترسی به دوربین در تنظیمات مرورگر شما فعال است.</p>
                            )}
                            {!detectedObjects.length && isDetecting && (
                                <p>هیچ شیء شناسایی نشد. از نور مناسب استفاده کنید یا تصویر دیگری امتحان کنید.</p>
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