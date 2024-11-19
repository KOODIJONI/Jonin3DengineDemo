// Window.h
#ifndef WINDOW_H
#define WINDOW_H
#include <string> 
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <strstream>
#include <vector> 
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace std;
class WindowApp {
public:
    // Constructor
    WindowApp(int width, int height, const char* title);
    
    // Destructor
    ~WindowApp();
   
    void setupLineShaders();

    GLuint lineVAO, lineVBO;
    bool onUserCreate();
    bool onUserUpdate(float fElapsedTime);
    GLuint shaderProgram;     // Main loop
    GLint colorLocation;
    void mainLoop();
    chrono::time_point<chrono::high_resolution_clock> startTime; 
    chrono::time_point<chrono::high_resolution_clock> lastTime;
    float getElapsedTime();
    int width, height;

    void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    struct Vector3 {
        float x, y, z, w;

        // Constructor
        Vector3(float x = 0, float y = 0, float z = 0, float w = 1) : x(x), y(y), z(z) , w(w) {}
    };

    struct mat4x4
    {
        float m[4][4] = { 0 };
    };
    struct Triangle { //kolmio



        Vector3 vertex1;
        Vector3 vertex2;
        Vector3 vertex3;
        Vector3 color;
        // Constructor alustamaan kolmion
        Triangle(const Vector3& v1, const Vector3& v2, const Vector3& v3,const Vector3& c)
            : vertex1(v1), vertex2(v2), vertex3(v3),color(c) {}
    };
    
    struct mesh {// objecti joka sis‰lt‰‰ 2d meshin tehtyn‰ kolmioista


        vector<Triangle> tris; //arrayn kaltainen "bufferi" joka sis‰lt‰‰ kolmiot
    };
    struct meshObject {
        vector<Triangle> tris;
        ifstream f; // Declare ifstream here as a member

        // Constructor to open the file
        meshObject(const std::string& filename) {
            f.open(filename); // Open the file in the constructor
            if (!f) {
                std::cerr << "Failed to open file: " << filename << std::endl;

            }
            else {
                vector<Vector3> verts;
                while (!f.eof())
                {
                    char line[128];
                    f.getline(line, 128);

                    strstream s;
                    s << line;

                    char junk;

                    if (line[0] == 'v' && line[1] != 'n')
                    {
                        Vector3 v;
                        s >> junk >> v.x >> v.y >> v.z;
                        verts.push_back(v);
                    }

                    if (line[0] == 'f')
                    {
                        int f[3];
                        s >> junk >> f[0] >> f[1] >> f[2];
                        try {
                            tris.push_back({ verts.at(f[0] - 1), verts.at(f[1] - 1), verts.at(f[2] - 1), Vector3(0,0,0) });
                        }
                        catch (const std::exception& e) {
                            std::cerr << "General exception: " << e.what() << std::endl;
                        }
                    }
                }

            }
        }
    };
    std::unique_ptr<meshObject> objPtr;

    // Function to initialize objPtr
    meshObject* initializeMeshObject(const std::string& filename);

private:

    float fTheta;
    
    void checkOpenGLError();
    void drawLine(WindowApp::Vector3& vertex1, WindowApp::Vector3& vertex2);
    void drawTriangle(Triangle& triProjected);
    float getColor(float dp);

    
    
    void MultiplyMatrixVector(Vector3& i, Vector3& o, mat4x4& m);
    mat4x4 matProj;
    Vector3 vCamera;
    Vector3 vLookDir;

    float fYaw;
    const char* title;
    GLFWwindow* window;

    // Initialize GLFW and create the window


    bool initGLFW();

    // Error callback
    static void errorCallback(int error, const char* description);
    Vector3 Vector_Add(Vector3& v1, Vector3& v2);
    Vector3 Vector_Sub(Vector3& v1, Vector3& v2);
    Vector3 Vector_Mul(Vector3& v1, float k);
    Vector3 Vector_Div(Vector3& v1, float k);
    float Vector_DotProduct(Vector3& v1, Vector3& v2);
    float Vector_Length(Vector3& v);
    Vector3 Vector_Normalise(Vector3& v);
    Vector3 Vector_CrossProduct(Vector3& v1, Vector3& v2);
    Vector3 Vector_IntersectPlane(Vector3& plane_p, Vector3& plane_n, Vector3& lineStart, Vector3& lineEnd);
    Vector3 Matrix_MultiplyVector(WindowApp::mat4x4& m, WindowApp::Vector3& i);

    mat4x4 Matrix_MakeIdentity();
    mat4x4 Matrix_MakeRotationX(float fAngleRad);
    mat4x4 Matrix_MakeRotationY(float fAngleRad);
    mat4x4 Matrix_MakeRotationZ(float fAngleRad);
    mat4x4 Matrix_MakeTranslation(float x, float y, float z);
    mat4x4 Matrix_MakeProjection(float fFovDegrees, float fAspectRatio, float fNear, float fFar);
    mat4x4 Matrix_MultiplyMatrix(mat4x4& m1, mat4x4& m2);
    mat4x4 Matrix_QuickInverse(mat4x4& m);
    mat4x4 Matrix_PointAt(Vector3& pos, Vector3& target, Vector3& up);
    /*WindowApp::Vector3 Vector_IntersectPlane(WindowApp::Vector3& plane_p, WindowApp::Vector3& plane_n, WindowApp::Vector3& lineStart, WindowApp::Vector3& lineEnd);
    */
};

#endif // WINDOW_H
