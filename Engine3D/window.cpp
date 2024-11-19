// Window.cpp
#include "Window.h"
#include <cmath>
#include <vector> 

// .cpp file
#include <algorithm>
#include <glad/glad.h>    
#include <GLFW/glfw3.h>  
#include <chrono>

using namespace std;

// Constructor
WindowApp::WindowApp(int width, int height, const char* title)

    : width(width), height(height), title(title), window(nullptr) {
    startTime = chrono::high_resolution_clock::now();
    lastTime = startTime;
    if (!initGLFW()) {
        std::cerr << "Failed to initialize GLFW and GLAD!" << std::endl;
        return;  
    }
    
    if (!onUserCreate()) {
        std::cerr << "User creation failed!" << std::endl;
        glfwTerminate(); 
        return;  
    }

    std::cout << "WindowApp and OpenGL context successfully initialized." << std::endl;

}
// Destructor
WindowApp::~WindowApp() {
    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}
// Main loop
void WindowApp::mainLoop() {
    if (!window) {
        std::cerr << "Window is not created properly!" << std::endl;
        return;
    }
    glViewport(0, 0, width, height);
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Process events
        glfwPollEvents();

        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //onUserUpdate();
        double currentTime = getElapsedTime();

        onUserUpdate(currentTime);
        
        

       

        checkOpenGLError();
        // Swap buffers
        glfwSwapBuffers(window);
    }

}
WindowApp::meshObject* WindowApp::initializeMeshObject(const std::string& filename) {
    objPtr = std::make_unique<meshObject>(filename); // Set objPtr to a new meshObject
    return objPtr.get(); // Return the pointer
}
float WindowApp:: getElapsedTime() {
    // Get the current time (current frame)
    auto currentTime = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time since the last frame
    std::chrono::duration<float> elapsed = currentTime - lastTime;

    // Update the lastTime for the next frame
    lastTime = currentTime;

    // Return the elapsed time in seconds
    return elapsed.count();
}
bool WindowApp::onUserCreate() {
    meshObject* objPtr = initializeMeshObject("C:/vsCodeCpp/pakettitesti/pakettitesti/kuutio.obj");
    float fNear = 0.1f;
    float fFar = 1000.0f;
    float fFov = 90.0f;
    float fAspectRatio = float(height) / float(width);
    float fFovRad =  1.0f / tanf(fFov * 0.5f / 180.0f * 3.14159);
    matProj.m[0][0] = fAspectRatio * fFovRad;
    matProj.m[1][1] = fFovRad;
    matProj.m[2][2] = fFar / (fFar-fNear);
    matProj.m[3][2] = (-fFar*fNear)  / (fFar - fNear);
    matProj.m[2][3] = 1.0f;
    matProj.m[3][3] = 0.0f;
    return true;
}
bool WindowApp::onUserUpdate(float fElapsedTime){
    Vector3 vForward = Vector_Mul(vLookDir, 1.00f);
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true); // Exit the loop if ESC is pressed
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        fYaw += 1* fElapsedTime;
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        vCamera = Vector_Add(vCamera, vForward);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        fYaw -= 1* fElapsedTime;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        vCamera = Vector_Sub(vCamera, vForward);
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        vCamera.y -= 100.0f* fElapsedTime;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        vCamera.y += 100.0f* fElapsedTime;
    }

   
   
    glClear(GL_COLOR_BUFFER_BIT);

    mat4x4 matRotX, matRotZ, matTrans, matWorld;
    mat4x4 matProj2 = Matrix_MakeProjection(90,width/height,1,1000);
    
    fTheta += 1.1f * fElapsedTime;
    matRotX = Matrix_MakeRotationX(fTheta);
    matRotZ = Matrix_MakeRotationZ(fTheta);
    matTrans = Matrix_MakeTranslation(0.0f, 0.0f, 150);
        // Rotation Z
    matWorld = Matrix_MultiplyMatrix(matRotZ, matRotX);
    matWorld = Matrix_MultiplyMatrix(matWorld, matTrans);

   
    Vector3 vUp = { 0,1,0 };
    Vector3 vTarget = { 0,0,1 };
    mat4x4 matCameraRot = Matrix_MakeRotationY(fYaw);
    vLookDir = Matrix_MultiplyVector(matCameraRot, vTarget);
    vTarget = Vector_Add(vCamera, vLookDir);
    mat4x4 matCamera = Matrix_PointAt(vCamera, vTarget, vUp);

    mat4x4 matView = Matrix_QuickInverse(matCamera);

    vector<Triangle> vecTrianglesToRaster;
    if (objPtr->tris.empty()) {
        std::cout << "tris is empty." << std::endl;
    }
    
    if (objPtr && !objPtr->tris.empty()) {
        for (auto tri : objPtr->tris) {

            Vector3 transformed1, transformed2, transformed3;
            Vector3 View1, View2, View3;
            
            transformed1 = Matrix_MultiplyVector(matWorld, tri.vertex1);
            transformed2 = Matrix_MultiplyVector(matWorld, tri.vertex2);
            transformed3 = Matrix_MultiplyVector(matWorld, tri.vertex3);
            

            Vector3 normal, line1, line2;

            line1 = Vector_Sub(transformed2, transformed1);
            line2 = Vector_Sub(transformed3, transformed1);

            normal = Vector_CrossProduct(line1, line2);
            normal = Vector_Normalise(normal);
            //std::cout << transformed1.x << transformed1.y << transformed1.z << std::endl;
      

            Vector3 vCameraRay = Vector_Sub(transformed1, vCamera);

            if (Vector_DotProduct(normal,vCameraRay)<0) {
                
                Vector3 lightDirection = { 0.0f, 0.0f, -1.0f };
                float dp = max(0.1f, Vector_DotProduct(lightDirection, normal));

                View1 = Matrix_MultiplyVector(matView, transformed1);
                View2 = Matrix_MultiplyVector(matView, transformed2);
                View3 = Matrix_MultiplyVector(matView, transformed3);



                Vector3 v1, v2, v3;
                v1 = Matrix_MultiplyVector(matProj2, View1);
                v2 = Matrix_MultiplyVector(matProj2, View2);
                v3 = Matrix_MultiplyVector(matProj2, View3);
                v1 = Vector3(v1.x / v1.w, v1.y / v1.w, v1.z / v1.w);
                v2 = Vector3(v2.x / v2.w, v2.y / v2.w, v2.z / v2.w);
                v3 = Vector3(v3.x / v3.w, v3.y / v3.w, v3.z / v3.w);
                float triangleColor = getColor(dp);/*
                std::cout << v1.x << v1.y << v1.z <<std::endl;
                std::cout << v2.x << v2.y << v2.z << std::endl;
                std::cout << v3.x << v3.y << v3.z << std::endl;*/

                Triangle triProjected(v1, v2, v3, Vector3(triangleColor, triangleColor, triangleColor));

                /*  triProjected.vertex1.x += 1.0f; triProjected.vertex1.y += 1.0f;
                  triProjected.vertex2.x += 1.0f; triProjected.vertex2.y += 1.0f;
                  triProjected.vertex3.x += 1.0f; triProjected.vertex3.y += 1.0f;


                  triProjected.vertex1.x = (triProjected.vertex1.x + 1.0f) * 0.5f ;
                  triProjected.vertex1.y = (1.0f - triProjected.vertex1.y) * 0.5f;
                  triProjected.vertex2.x = (triProjected.vertex2.x + 1.0f) * 0.5f;
                  triProjected.vertex2.y = (1.0f - triProjected.vertex2.y) * 0.5f;
                  triProjected.vertex3.x = (triProjected.vertex3.x + 1.0f) * 0.5f;
                  triProjected.vertex3.y = (1.0f - triProjected.vertex3.y) * 0.5f ;
                  */
                vecTrianglesToRaster.push_back(triProjected);
              
            }

        }

    }
    std::sort(vecTrianglesToRaster.begin(), vecTrianglesToRaster.end(), [](Triangle& t1, Triangle& t2)
        {
            float z1 = (t1.vertex1.z + t1.vertex2.z + t1.vertex3.z) / 3.0f;
            float z2 = (t2.vertex1.z + t2.vertex2.z + t2.vertex3.z) / 3.0f;
            return z1 > z2;
        });

    for (auto& triProjected : vecTrianglesToRaster)
    {
        // Rasterize triangle
        drawTriangle(triProjected);
    }
    vecTrianglesToRaster.clear();

    return true;
}

// Initialize GLFW and create the window
bool WindowApp::initGLFW() {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {




        return false;
    }

    // Set window hints for OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create the window
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return false;
    }

    // Make the OpenGL context current
    glfwMakeContextCurrent(window);
    glfwSetWindowSizeCallback(window, [](GLFWwindow* window, int newWidth, int newHeight)
        {
            WindowApp* app = reinterpret_cast<WindowApp*>(glfwGetWindowUserPointer(window));
            if (app) {
                // Update the width and height of the object
                app->width = newWidth;
                app->height = newHeight;

                // Update the OpenGL viewport
                glViewport(0, 0, app->width, app->height);
            }
        });

    glfwSetWindowUserPointer(window, this);
    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    setupLineShaders(); // Initialize shaders and buffers here, as GLAD is now ready
   

    return true;
}
// Error callback
void WindowApp::errorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}
void WindowApp::checkOpenGLError() {
    GLenum error = glGetError();
    while (error != GL_NO_ERROR) {
        switch (error) {
        case GL_INVALID_ENUM:
            std::cerr << "GL_INVALID_ENUM: An unacceptable value is specified for an enumerated argument." << std::endl;
            break;
        case GL_INVALID_VALUE:
            std::cerr << "GL_INVALID_VALUE: A numeric argument is out of range." << std::endl;
            break;
        case GL_INVALID_OPERATION:
            std::cerr << "GL_INVALID_OPERATION: The specified operation is not allowed in the current state." << std::endl;
            break;
        case GL_OUT_OF_MEMORY:
            std::cerr << "GL_OUT_OF_MEMORY: There is not enough memory left to execute the command." << std::endl;
            break;
        default:
            std::cerr << "Unknown OpenGL error!" << std::endl;
            break;
        }
        error = glGetError(); // Check for the next error
    }
}
void WindowApp::setupLineShaders() {
    // Vertex shader for the line
    const char* vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
        })";

    // Fragment shader for the line (simple color)
    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec4 objectColor;
        void main() {
            FragColor = objectColor;  // Red color
        })";

    // Compile shaders and link them to a shader program
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    // Check for compilation errors
    GLint success;
    GLchar infoLog[512];

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    // Check for compilation errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // Create VAO and VBO for line drawing
    glGenVertexArrays(1, &lineVAO);
    glGenBuffers(1, &lineVBO);
}
// Function to draw a line between points (x1, y1) and (x2, y2)
void WindowApp::drawLine(WindowApp::Vector3& vertex1, WindowApp::Vector3& vertex2) {
    // Define the line's vertex positions (two points)
    GLfloat lineVertices[] = {
        vertex1.x, vertex1.y,  // Point A (x1, y1)
        vertex2.x, vertex2.y  // Point B (x2, y2)
    };

    // Bind the VAO and VBO to load the vertex data
    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(lineVertices), lineVertices, GL_STATIC_DRAW);

    // Define vertex attributes (positions)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // Use the shader program for rendering
    glUseProgram(shaderProgram);

    // Draw the line (GL_LINES mode)
    glDrawArrays(GL_LINES, 0, 2);

    // Unbind the VAO and VBO after drawing
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
void WindowApp::drawTriangle(Triangle& triangle) {
    // Define the triangle's vertex positions (three points)
    GLfloat triangleVertices[] = {
        triangle.vertex1.x, triangle.vertex1.y,  // Point 1 (x1, y1)
        triangle.vertex2.x, triangle.vertex2.y,  // Point 2 (x2, y2)
        triangle.vertex3.x, triangle.vertex3.y   // Point 3 (x3, y3)
    };

    // Bind the VAO and VBO to load the vertex data
    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);

    // Define vertex attributes (positions)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // Use the shader program for rendering
    glUseProgram(shaderProgram);
    GLint colorLocation = glGetUniformLocation(shaderProgram, "objectColor");
    if (colorLocation == -1) {
        std::cout << "ERROR::UNIFORM::OBJECT_COLOR_NOT_FOUND" << std::endl;
    }
    glUniform4f(colorLocation, triangle.color.x, triangle.color.y, triangle.color.z, 1.0f);
    // Draw the triangle (GL_TRIANGLES mode)
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Unbind the VAO and VBO after drawing
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
float WindowApp::getColor(float dp) {
    return std::max(0.0f, std::min(1.0f, dp));
}
void WindowApp::MultiplyMatrixVector(Vector3& i, Vector3& o, mat4x4& m) {
    o.x = i.x * m.m[0][0] + i.y * m.m[1][0] + i.z * m.m[2][0] + m.m[3][0];
    o.y = i.x * m.m[0][1] + i.y * m.m[1][1] + i.z * m.m[2][1] + m.m[3][1];
    o.z = i.x * m.m[0][2] + i.y * m.m[1][2] + i.z * m.m[2][2] + m.m[3][2];
    float w = i.x * m.m[0][3] + i.y * m.m[1][3] + i.z * m.m[2][3] + m.m[3][3];

    if (w != 0.0f) {
        o.x /= w; o.y /= w; o.z /= w;
    }
}
WindowApp::Vector3 WindowApp::Matrix_MultiplyVector(WindowApp::mat4x4& m, WindowApp::Vector3& i)
{
    WindowApp::Vector3 v;
    v.x = i.x * m.m[0][0] + i.y * m.m[1][0] + i.z * m.m[2][0] + i.w * m.m[3][0];
    v.y = i.x * m.m[0][1] + i.y * m.m[1][1] + i.z * m.m[2][1] + i.w * m.m[3][1];
    v.z = i.x * m.m[0][2] + i.y * m.m[1][2] + i.z * m.m[2][2] + i.w * m.m[3][2];
    v.w = i.x * m.m[0][3] + i.y * m.m[1][3] + i.z * m.m[2][3] + i.w * m.m[3][3];
    return v;
}
WindowApp::Vector3 WindowApp::Vector_Add(WindowApp::Vector3& v1, WindowApp::Vector3& v2)
{
    return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}

WindowApp::Vector3 WindowApp::Vector_Sub(WindowApp::Vector3& v1, WindowApp::Vector3& v2)
{
    return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}

WindowApp::Vector3 WindowApp::Vector_Mul(WindowApp::Vector3& v1, float k)
{
    return { v1.x * k, v1.y * k, v1.z * k };
}

WindowApp::Vector3 WindowApp::Vector_Div(WindowApp::Vector3& v1, float k)
{
    return { v1.x / k, v1.y / k, v1.z / k };
}

float WindowApp::Vector_DotProduct(WindowApp::Vector3& v1, WindowApp::Vector3& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

float WindowApp::Vector_Length(WindowApp::Vector3& v)
{
    return sqrtf(Vector_DotProduct(v, v));
}

WindowApp::Vector3 WindowApp::Vector_Normalise(WindowApp::Vector3& v)
{
    float l = Vector_Length(v);
    return { v.x / l, v.y / l, v.z / l };
}

WindowApp::Vector3 WindowApp::Vector_CrossProduct(WindowApp::Vector3& v1, WindowApp::Vector3& v2)
{
    WindowApp::Vector3 v;
    v.x = v1.y * v2.z - v1.z * v2.y;
    v.y = v1.z * v2.x - v1.x * v2.z;
    v.z = v1.x * v2.y - v1.y * v2.x;
    return v;
}

WindowApp::Vector3 WindowApp::Vector_IntersectPlane(WindowApp::Vector3& plane_p, WindowApp::Vector3& plane_n, WindowApp::Vector3& lineStart, WindowApp::Vector3& lineEnd)
{
    plane_n = Vector_Normalise(plane_n);
    float plane_d = -Vector_DotProduct(plane_n, plane_p);
    float ad = Vector_DotProduct(lineStart, plane_n);
    float bd = Vector_DotProduct(lineEnd, plane_n);
    float t = (-plane_d - ad) / (bd - ad);
    WindowApp::Vector3 lineStartToEnd = Vector_Sub(lineEnd, lineStart);
    WindowApp::Vector3 lineToIntersect = Vector_Mul(lineStartToEnd, t);
    return Vector_Add(lineStart, lineToIntersect);
}

WindowApp::mat4x4 WindowApp::Matrix_MakeIdentity()
{
    WindowApp::mat4x4 matrix;
    matrix.m[0][0] = 1.0f;
    matrix.m[1][1] = 1.0f;
    matrix.m[2][2] = 1.0f;
    matrix.m[3][3] = 1.0f;
    return matrix;
}
WindowApp::mat4x4 WindowApp::Matrix_PointAt(WindowApp::Vector3& pos, WindowApp::Vector3& target, WindowApp::Vector3& up) {
    Vector3 newForward = Vector_Sub(target, pos);
    newForward = Vector_Normalise(newForward);

    Vector3 a = Vector_Mul(newForward, Vector_DotProduct(up, newForward));
    Vector3 newUp = Vector_Sub(up, a);
    newUp = Vector_Normalise(newUp);

    Vector3 newRight = Vector_CrossProduct(newUp, newForward);
    mat4x4 matrix;
    matrix.m[0][0] = newRight.x;	matrix.m[0][1] = newRight.y;	matrix.m[0][2] = newRight.z;	matrix.m[0][3] = 0.0f;
    matrix.m[1][0] = newUp.x;		matrix.m[1][1] = newUp.y;		matrix.m[1][2] = newUp.z;		matrix.m[1][3] = 0.0f;
    matrix.m[2][0] = newForward.x;	matrix.m[2][1] = newForward.y;	matrix.m[2][2] = newForward.z;	matrix.m[2][3] = 0.0f;
    matrix.m[3][0] = pos.x;			matrix.m[3][1] = pos.y;			matrix.m[3][2] = pos.z;			matrix.m[3][3] = 1.0f;
    return matrix;
}
WindowApp::mat4x4 WindowApp::Matrix_QuickInverse(WindowApp::mat4x4& m) // Only for Rotation/Translation Matrices
{
    mat4x4 matrix;
    matrix.m[0][0] = m.m[0][0]; matrix.m[0][1] = m.m[1][0]; matrix.m[0][2] = m.m[2][0]; matrix.m[0][3] = 0.0f;
    matrix.m[1][0] = m.m[0][1]; matrix.m[1][1] = m.m[1][1]; matrix.m[1][2] = m.m[2][1]; matrix.m[1][3] = 0.0f;
    matrix.m[2][0] = m.m[0][2]; matrix.m[2][1] = m.m[1][2]; matrix.m[2][2] = m.m[2][2]; matrix.m[2][3] = 0.0f;
    matrix.m[3][0] = -(m.m[3][0] * matrix.m[0][0] + m.m[3][1] * matrix.m[1][0] + m.m[3][2] * matrix.m[2][0]);
    matrix.m[3][1] = -(m.m[3][0] * matrix.m[0][1] + m.m[3][1] * matrix.m[1][1] + m.m[3][2] * matrix.m[2][1]);
    matrix.m[3][2] = -(m.m[3][0] * matrix.m[0][2] + m.m[3][1] * matrix.m[1][2] + m.m[3][2] * matrix.m[2][2]);
    matrix.m[3][3] = 1.0f;
    return matrix;
}

WindowApp::mat4x4 WindowApp::Matrix_MakeRotationX(float fAngleRad)
{
    WindowApp::mat4x4 matrix;
    matrix.m[0][0] = 1.0f;
    matrix.m[1][1] = cosf(fAngleRad);
    matrix.m[1][2] = sinf(fAngleRad);
    matrix.m[2][1] = -sinf(fAngleRad);
    matrix.m[2][2] = cosf(fAngleRad);
    matrix.m[3][3] = 1.0f;
    return matrix;
}

WindowApp::mat4x4 WindowApp::Matrix_MakeRotationY(float fAngleRad)
{
    WindowApp::mat4x4 matrix;
    matrix.m[0][0] = cosf(fAngleRad);
    matrix.m[0][2] = sinf(fAngleRad);
    matrix.m[2][0] = -sinf(fAngleRad);
    matrix.m[1][1] = 1.0f;
    matrix.m[2][2] = cosf(fAngleRad);
    matrix.m[3][3] = 1.0f;
    return matrix;
}

WindowApp::mat4x4 WindowApp::Matrix_MakeRotationZ(float fAngleRad)
{
    WindowApp::mat4x4 matrix;
    matrix.m[0][0] = cosf(fAngleRad);
    matrix.m[0][1] = sinf(fAngleRad);
    matrix.m[1][0] = -sinf(fAngleRad);
    matrix.m[1][1] = cosf(fAngleRad);
    matrix.m[2][2] = 1.0f;
    matrix.m[3][3] = 1.0f;
    return matrix;
}

WindowApp::mat4x4 WindowApp::Matrix_MakeTranslation(float x, float y, float z)
{
    WindowApp::mat4x4 matrix;
    matrix.m[0][0] = 1.0f;
    matrix.m[1][1] = 1.0f;
    matrix.m[2][2] = 1.0f;
    matrix.m[3][3] = 1.0f;
    matrix.m[3][0] = x;
    matrix.m[3][1] = y;
    matrix.m[3][2] = z;
    return matrix;
}

WindowApp::mat4x4 WindowApp::Matrix_MakeProjection(float fFovDegrees, float fAspectRatio, float fNear, float fFar)
{
    float fFovRad = 1.0f / tanf(fFovDegrees * 0.5f / 180.0f * 3.14159f);
    WindowApp::mat4x4 matrix;
    matrix.m[0][0] = fAspectRatio * fFovRad;
    matrix.m[1][1] = fFovRad;
    matrix.m[2][2] = fFar / (fFar - fNear);
    matrix.m[3][2] = (-fFar * fNear) / (fFar - fNear);
    matrix.m[2][3] = 1.0f;
    matrix.m[3][3] = 0.0f;
    return matrix;
}

WindowApp::mat4x4 WindowApp::Matrix_MultiplyMatrix(WindowApp::mat4x4& m1, WindowApp::mat4x4& m2)
{
    WindowApp::mat4x4 matrix;
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            matrix.m[r][c] = m1.m[r][0] * m2.m[0][c] + m1.m[r][1] * m2.m[1][c] + m1.m[r][2] * m2.m[2][c] + m1.m[r][3] * m2.m[3][c];
    return matrix;
}/*
WindowApp::Vector3 WindowApp::Vector_IntersectPlane(WindowApp::Vector3& plane_p, WindowApp::Vector3& plane_n, WindowApp::Vector3& lineStart, WindowApp::Vector3& lineEnd)
{
    plane_n = Vector_Normalise(plane_n);
    float plane_d = -Vector_DotProduct(plane_n, plane_p);
    float ad = Vector_DotProduct(lineStart, plane_n);
    float bd = Vector_DotProduct(lineEnd, plane_n);
    float t = (-plane_d - ad) / (bd - ad);
    WindowApp::Vector3 lineStartToEnd = Vector_Sub(lineEnd, lineStart);
    WindowApp::Vector3 lineToIntersect = Vector_Mul(lineStartToEnd, t);
    return Vector_Add(lineStart, lineToIntersect);
}
*/