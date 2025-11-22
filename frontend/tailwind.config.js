/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'page-bg': '#EDF6F7',
                'card-bg': '#F8F7FE',
                'accent': '#E2E8F0', // Slightly darker for borders if needed, or keep consistent
            },
        },
    },
    plugins: [],
}
