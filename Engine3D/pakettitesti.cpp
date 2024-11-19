#include <stdio.h>
#include "Window.h"


int main() {
    WindowApp window(800, 600, "Projekti ikkuna");
    window.mainLoop();
    return 0;

}