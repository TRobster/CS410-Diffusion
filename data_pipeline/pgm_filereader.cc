#include <utility>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstdint>

using namespace std;

// Advance past ASCII whitespace and PGM comment lines (lines starting with '#').
static void skip_comments(ifstream& f)
{
    char c;
    while (f.peek() != EOF) {
        if (isspace((unsigned char)f.peek())) {
            f.get(c);
        } else if (f.peek() == '#') {
            while (f.get(c) && c != '\n') {}
        } else {
            break;
        }
    }
}

// Returns pair of rows, cols
pair<int,int> rows_and_cols(string filename)
{
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        cerr << "pgm_filereader: cannot open \"" << filename << "\"\n";
        exit(EXIT_FAILURE);
    }

    string magic;
    f >> magic;
    if (magic != "P2" && magic != "P5") {
        cerr << "pgm_filereader: unsupported format \"" << magic
             << "\" (expected P2 or P5)\n";
        exit(EXIT_FAILURE);
    }

    skip_comments(f);
    int cols, rows;
    f >> cols >> rows;

    return {rows, cols};
}

vector<float> read_pgm_file(string filename, bool row_major)
{
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        cerr << "pgm_filereader: cannot open \"" << filename << "\"\n";
        exit(EXIT_FAILURE);
    }

    string magic;
    f >> magic;
    if (magic != "P2" && magic != "P5") {
        cerr << "pgm_filereader: unsupported format \"" << magic
             << "\" (expected P2 or P5)\n";
        exit(EXIT_FAILURE);
    }

    skip_comments(f);
    int cols, rows, maxval;
    f >> cols >> rows;
    skip_comments(f);
    f >> maxval;

    const int n = rows * cols;
    vector<float> pixels(n);

    if (magic == "P2") {
        // ASCII: one integer per pixel
        for (int i = 0; i < n; i++) {
            int v;
            f >> v;
            pixels[i] = (float)v / maxval;
        }
    } else {
        // P5 binary: one whitespace byte follows maxval, then raw pixel data
        char ws;
        f.get(ws);

        if (maxval < 256) {
            vector<uint8_t> buf(n);
            f.read(reinterpret_cast<char*>(buf.data()), n);
            for (int i = 0; i < n; i++)
                pixels[i] = (float)buf[i] / maxval;
        } else {
            // 16-bit big-endian P5
            vector<uint8_t> buf(n * 2);
            f.read(reinterpret_cast<char*>(buf.data()), n * 2);
            for (int i = 0; i < n; i++) {
                uint16_t v = ((uint16_t)buf[2*i] << 8) | buf[2*i + 1];
                pixels[i] = (float)v / maxval;
            }
        }
    }

    if (row_major)
        return pixels;

    // Transpose to column-major: result[col * rows + row]
    vector<float> col_major(n);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            col_major[c * rows + r] = pixels[r * cols + c];
    return col_major;
}
