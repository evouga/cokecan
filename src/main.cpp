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

int numSteps;
double tol;

void lameParameters(double& alpha, double& beta)
{
    double young = 1.0 / thickness; // doesn't matter for static solves
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

void runSimulation(
    const LibShell::MeshConnectivity& mesh,
    Eigen::MatrixXd& curPos,
    double crushRatio,
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
    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameAlpha);

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

    NeohookeanShellEnergy energyModel(mesh, restState);
    //QuadraticBendingShellEnergy energyModel(mesh, restState, restPos, restEdgeDOFs);

    double reg = 1e-6;
    for (int j = 1; j <= numSteps; j++)
    {
        double prevt = double(j - 1) / double(numSteps);
        double t = double(j) / double(numSteps);
        double curRatio = t * crushRatio + (1 - t);
        double prevRatio = prevt * crushRatio + (1 - prevt);
        double curHeight = cokeHeight * curRatio;

        // uniformly crush everything
        for (int i = 0; i < curPos.rows(); i++)
            curPos(i, 2) *= curRatio / prevRatio;

        // pin boundaries just in case
        for (int i : topVertices)
            curPos(i, 2) = curHeight;
        for (int i : bottomVertices)
            curPos(i, 2) = 0;

        takeOneStep(energyModel, curPos, edgeDOFs, fixed, tol, reg);
        std::stringstream filename;
        filename << "step-" << j << ".ply";
        igl::writePLY(filename.str(), curPos, mesh.faces());
        std::cout << "####################" << std::endl;
        std::cout << "Finished Step " << j << std::endl;
        std::cout << "####################" << std::endl;
        polyscope::registerSurfaceMesh(filename.str(), curPos, mesh.faces());
    }
}

int main(int argc, char* argv[])
{
    numSteps = 10;
    tol = 1e-8;

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


        if (ImGui::CollapsingHeader("Optimization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputInt("Num Steps", &numSteps);
            ImGui::InputDouble("Newton Tolerance", &tol);
            if (ImGui::Button("Crush", ImVec2(-1, 0)))
            {
                double lameAlpha, lameBeta;
                lameParameters(lameAlpha, lameBeta);
                Eigen::MatrixXd curPos = origV;
                // set up mesh connectivity
                LibShell::MeshConnectivity mesh(F);
                runSimulation(mesh, curPos, crushAmount, thickness, lameAlpha, lameBeta);
                polyscope::registerSurfaceMesh("Crushed Result", curPos, F);
            }
        }
    };

    polyscope::show();
}
