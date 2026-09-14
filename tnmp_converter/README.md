# Texture → Normal Map Generator

An AI-powered web application that generates **realistic normal maps from texture images** using a trained **Pix2Pix-style U-Net generator** and **PyTorch**.

The application provides a simple Flask web interface where users can upload a texture, generate its corresponding normal map, preview the result, download it, and browse previously generated maps.

---

## ✨ Features

* 🖼️ Upload texture images
* 🤖 Generate normal maps using a trained PyTorch model
* 🧠 Pix2Pix-style U-Net Generator architecture
* ⚡ Automatic GPU/CPU detection
* 💾 Save uploaded textures and generated normal maps
* 🗄️ Store generation history using SQLite
* 🆔 Generate a unique UUID for every generation
* 📥 Download generated normal maps
* 🖼️ Gallery showing previous texture → normal map generations
* 📱 Responsive web interface
* 🔒 Secure uploaded filenames using `secure_filename`

