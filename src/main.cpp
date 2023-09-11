#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "StaticSolve.h"
#include "Cylinder.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"
#include "igl/readOBJ.h"
#include <set>
#include <vector>
#include "ShellEnergy.h"
#include "igl/writePLY.h"

double cokeRadius;
double cokeHeight;

double thickness;
double poisson;

double crushAmount;

double triangleArea;

float rotAxis[3];
int rotSteps;

void lameParameters(double& alpha, double& beta)
{
    double young = 1.0 / thickness; // doesn't matter for static solves
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

void testInvariance(
    const LibShell::MeshConnectivity& mesh,
    Eigen::MatrixXd& curPos,
    Eigen::Vector3d rotAxis,
    int rotSteps,
    double thickness,
    double lameAlpha,
    double lameBeta)
{
    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edgeDOFs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edgeDOFs, mesh, curPos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState restState;

    // set uniform thicknesses
    restState.thicknesses.resize(mesh.nFaces(), thickness);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::firstFundamentalForms(mesh, curPos, restState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, curPos, edgeDOFs, restState.bbars);

    std::vector<int> topVertices;
    std::vector<int> bottomVertices;
    getBoundaries(curPos, mesh.faces(), topVertices, bottomVertices);

    std::set<int> fixed;
    for (int i : topVertices)
        fixed.insert(i);
    for (int i : bottomVertices)
        fixed.insert(i);

    Eigen::MatrixXd restPos = curPos;
    Eigen::VectorXd restEdgeDOFs = edgeDOFs;

    //NeohookeanShellEnergy energyModel(mesh, restState, lameAlpha, lameBeta);
    QuadraticBendingShellEnergy energyModel(mesh, restState, restPos, restEdgeDOFs, lameAlpha, lameBeta);

    rotAxis /= rotAxis.norm();
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(2.0 * 3.1415926535898 / rotSteps, rotAxis);


    std::cout << "Begin experiment, " << rotSteps << " steps about axis " << rotAxis.transpose() << std::endl;
    for (int j = 1; j <= rotSteps; j++)
    {
        // rotate mesh
        curPos = curPos * R.transpose();
        double E = energyModel.elasticEnergy(curPos, edgeDOFs, NULL, NULL);
        std::cout << E << std::endl;
    }
    std::cout << "End experiment" << std::endl;
}

int main(int argc, char* argv[])
{
    rotAxis[0] = 1.0;
    rotAxis[1] = 1.0;
    rotAxis[2] = 1.0;
    rotSteps = 100;

    cokeRadius = 0.0325;
    cokeHeight = 0.122;

    crushAmount = 0.95;

    triangleArea = 0.000001;


    // set up material parameters
    thickness = 0.000102;
    poisson = 1.0 / 2.0;

    // load mesh

    Eigen::MatrixXd origV;
    Eigen::MatrixXi F;

    makeCylinder(cokeRadius, cokeHeight, triangleArea, origV, F);

    polyscope::init();
    polyscope::registerSurfaceMesh("Input Mesh", origV, F);
    polyscope::state::userCallback = [&]()
    {
        if (ImGui::CollapsingHeader("Geometry", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputDouble("Radius", &cokeRadius);
            ImGui::InputDouble("Height", &cokeHeight);
            ImGui::InputDouble("Triangle Area", &triangleArea);
            if (ImGui::Button("Retriangulate"))
            {
                makeCylinder(cokeRadius, cokeHeight, triangleArea, origV, F);                
            }
        }
        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputDouble("Thickness", &thickness);
            ImGui::InputDouble("Poisson's Ration", &poisson);
            ImGui::InputDouble("Crush Ratio", &crushAmount);
        }


        if (ImGui::CollapsingHeader("Test", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputFloat3("Rotation Axis", rotAxis, 4);
            ImGui::InputInt("Rotation Steps", &rotSteps);
            if (ImGui::Button("Test Rotation-invariance", ImVec2(0 - 1, 0)))
            {
                double lameAlpha, lameBeta;
                lameParameters(lameAlpha, lameBeta);
                Eigen::MatrixXd curPos = origV;
                polyscope::registerSurfaceMesh("Perturbed Mesh", curPos, F);

                // set up mesh connectivity
                LibShell::MeshConnectivity mesh(F);
                testInvariance(mesh, curPos, Eigen::Vector3d(rotAxis[0], rotAxis[1], rotAxis[2]), rotSteps, thickness, lameAlpha, lameBeta);
                
            }
        }
    };

    polyscope::show();
}
