# Sock: A really fast Sudoku solver written in Rust

**Sock** is an ultra-fast Sudoku puzzle solver built using the power and safety of Rust.

- **Speed**: Built with performance in mind. FastSudoku harnesses the speed of Rust's concurrency mechanisms and optimized algorithms.
  
- **Cross-Platform**: Works on MacOS, Linux, and Windows.

- **Open Source**: Dive into the code and contribute!

---

## 🛠 Build Instructions

1. **Prerequisites**
   
   Ensure you have Rust and Cargo installed. If not, install them using [rustup](https://rustup.rs/).

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Clone the Repository**
   
   ```bash
   git clone --recursive https://github.com/Crypec/sock.git
   cd sock
   ```
3. **Unpack Test Data**
```bash
unzip test_data.zip 
```

4. **Build the Project**

   ```bash
   cargo build --release
   ```

## 🤖 Usage

```rust
let mut solver = Solver::new(board);
let res = solver.solve(); 
```


---

## 🤝 Contribute

Feel free to dive into the project. Pull requests and contributions are welcomed!

1. Fork the repository
2. Clone your forked repository
3. Create a feature branch (`git checkout -b feature/YourFeature`)
4. Commit your changes
5. Push to the branch
6. Open a Pull Request

---

## 📜 License

Currently not decided.
---

🌟 Don't forget to star the repository if you find it useful! 🌟
