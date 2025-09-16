//@ts-nocheck
import { motion } from 'framer-motion';
import React from 'react';

const LoadingAnimation: React.FC = () => {
    const svgVariants = {
        animate: {
            rotate: [0, 360],
            transition: {
                delay: 1.5, // بعد از اتمام حرکت به مرکز
                duration: 2.5,
                repeat: Infinity,
                ease: 'linear',
            },
        },
    };

    const pathVariants = {
        animate: (i: number) => ({
            scale: [1.5, 1],
            x: [
                i === 0 ? '70px' : i === 1 ? '0px' : i === 2 ? '-70px' : '0px',
                '0px',
            ],
            y: [
                i === 0 ? '0px' : i === 1 ? '70px' : i === 2 ? '0px' : '-70px',
                '0px',
            ],
            transition: {
                scale: { duration: 1.5, ease: 'easeInOut' },
                x: { duration: 1.5, ease: 'easeInOut' },
                y: { duration: 1.5, ease: 'easeInOut' },
            },
        }),
    };

    return (
        <div className="absolute inset-0 flex items-center justify-center ">
            <svg
                width="70"
                height="70"
                viewBox="0 0 140 141"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="relative"
            >
                <path
                    d="M17.8002 140.015H122.2C126.921 140.013 131.449 138.136 134.787 134.797C138.125 131.457 140 126.929 140 122.207V17.8076C140 13.086 138.125 8.55772 134.787 5.21836C131.449 1.879 126.921 0.00197721 122.2 0H17.8002C15.4598 0.000976789 13.1425 0.463299 10.9809 1.36052C8.81929 2.25774 6.85575 3.57226 5.20256 5.22891C3.54937 6.88556 2.23896 8.85184 1.34627 11.0153C0.453572 13.1788 -0.00389551 15.497 2.49868e-05 17.8374V122.237C2.45729e-05 126.959 1.87515 131.487 5.21312 134.826C8.55108 138.166 13.0786 140.043 17.8002 140.045"
                    fill="#FFC20E90"
                />
            </svg>
            <motion.svg
                width="70"
                height="70"
                viewBox="0 0 140 141"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="absolute"
                variants={svgVariants}
                animate="animate"
                style={{ originX: '50%', originY: '50%' }} // مرکز SVG برای چرخش
            >
                <motion.path
                    d="M122.558 51.7748L93.0873 81.2453H65.8613L95.3319 51.7748L84.7577 41.2081V13.9821L122.558 51.7748Z"
                    fill="#073054"
                    variants={pathVariants}
                    animate="animate"
                    custom={0}
                />
                <motion.path
                    d="M124.586 117.942H82.9083L63.6541 98.695H105.332V83.7435L124.586 64.4966V117.942Z"
                    fill="#073054"
                    variants={pathVariants}
                    animate="animate"
                    custom={1}
                />
                <motion.path
                    d="M17.4199 88.2699L46.8905 58.7994H74.1164L44.6459 88.2699L55.2126 98.8367V126.063L17.4199 88.2699Z"
                    fill="#073054"
                    variants={pathVariants}
                    animate="animate"
                    custom={2}
                />
                <motion.path
                    d="M15.3916 22.1029H57.0695L76.3163 41.3497H34.6384V56.3013L15.3916 75.5481V22.1029Z"
                    fill="#073054"
                    variants={pathVariants}
                    animate="animate"
                    custom={3}
                />
            </motion.svg>
        </div>
    );
};

export default LoadingAnimation;