/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx}"] ,
  theme: {
    extend: {
      fontFamily: {
        sans: ["Plus Jakarta Sans", "system-ui", "sans-serif"],
        mono: ["Fira Code", "ui-monospace", "monospace"]
      },
      colors: {
        base: {
          900: "#0f1115",
          800: "#171a21",
          700: "#202532",
          600: "#2f3646",
          200: "#d6d9e0",
          100: "#eef0f5"
        }
      }
    }
  },
  plugins: []
};
