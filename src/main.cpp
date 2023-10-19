#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "StaticSolve.h"
#include "HalfCylinder.h"
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
#include "polyscope/surface_vector_quantity.h"

double cokeRadius;
double cokeHeight;

double thickness;
double poisson;

double triangleArea;

double nhShellEnergy;
double qbShellEnergy;
double QBEnergy;

void lameParameters(double& alpha, double& beta)
{
    double young = 1.0 / thickness; // doesn't matter for static solves
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

std::pair<double, double> measureEnergy(
    const LibShell::MeshConnectivity& mesh,
    Eigen::MatrixXd& curPos,
    double thickness,
    double lameAlpha,
    double lameBeta,
    Eigen::MatrixXd &nhForces,
    Eigen::MatrixXd &qbForces)
{
    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edgeDOFs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edgeDOFs, mesh, curPos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState restState;

    // set uniform thicknesses
    restState.thicknesses.resize(mesh.nFaces(), thickness);
    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::firstFundamentalForms(mesh, curPos, restState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, curPos, edgeDOFs, restState.bbars);

    // Make the half-cylinder rest-flat
    for (int i = 0; i < mesh.nFaces(); i++)
        restState.bbars[i].setZero();

    Eigen::MatrixXd restPos = curPos;
    Eigen::VectorXd restEdgeDOFs = edgeDOFs;

    NeohookeanShellEnergy nhenergyModel(mesh, restState);
    QuadraticBendingShellEnergy qbenergyModel(mesh, restState, restPos, restEdgeDOFs);

    Eigen::VectorXd nhF;
    Eigen::VectorXd qbF;
    double nhenergy = nhenergyModel.elasticEnergy(curPos, edgeDOFs, &nhF, NULL);
    double qbenergy = qbenergyModel.elasticEnergy(curPos, edgeDOFs, &qbF, NULL);

    int nverts = curPos.rows();
    nhForces.resize(nverts, 3);
    qbForces.resize(nverts, 3);
    for (int i = 0; i < nverts; i++)
    {
        nhForces.row(i) = -nhF.segment<3>(3 * i);
        qbForces.row(i) = -qbF.segment<3>(3 * i);
    }
    return { nhenergy, qbenergy };
}

int main(int argc, char* argv[])
{
    cokeRadius = 0.0325;
    cokeHeight = 0.122;

    triangleArea = 0.000001;

    nhShellEnergy = 0;
    qbShellEnergy = 0;
    QBEnergy = 0;

    Eigen::MatrixXd nhForces;
    Eigen::MatrixXd qbForces;


    // set up material parameters
    thickness = 0.00010;
    poisson = 1.0 / 2.0;

    // load mesh

    Eigen::MatrixXd origV;
    Eigen::MatrixXi F;

    makeHalfCylinder(cokeRadius, cokeHeight, triangleArea, origV, F);
    double lameAlpha, lameBeta;
    lameParameters(lameAlpha, lameBeta);
    LibShell::MeshConnectivity mesh(F);
    auto energies = measureEnergy(mesh, origV, thickness, lameAlpha, lameBeta, nhForces, qbForces);
    nhShellEnergy = energies.first;
    qbShellEnergy = energies.second;

    polyscope::init();
    auto *surf = polyscope::registerSurfaceMesh("Input Mesh", origV, F);

    surf->addVertexVectorQuantity("NH Force", nhForces);
    surf->addVertexVectorQuantity("QB Force", qbForces);

    polyscope::state::userCallback = [&]()
    {
        bool dirty = false;
        if (ImGui::CollapsingHeader("Geometry", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputDouble("Radius", &cokeRadius);
            ImGui::InputDouble("Height", &cokeHeight);
            ImGui::InputDouble("Triangle Area", &triangleArea);
            if (ImGui::Button("Retriangulate"))
            {
                makeHalfCylinder(cokeRadius, cokeHeight, triangleArea, origV, F);   
                surf = polyscope::registerSurfaceMesh("Input Mesh", origV, F);
                dirty = true;                
            }
        }
        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if(ImGui::InputDouble("Thickness", &thickness))
                dirty = true;
            if(ImGui::InputDouble("Poisson's Ration", &poisson))
                dirty = true;            
        }

        if (dirty)
        {
            mesh = LibShell::MeshConnectivity(F);
            lameParameters(lameAlpha, lameBeta);
            auto energies = measureEnergy(mesh, origV, thickness, lameAlpha, lameBeta, nhForces, qbForces);
            nhShellEnergy = energies.first;
            qbShellEnergy = energies.second;
            surf->addVertexVectorQuantity("NH Force", nhForces);
            surf->addVertexVectorQuantity("QB Force", qbForces);
        }        

        if (ImGui::CollapsingHeader("Energies", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Neohookean Shell: %e", nhShellEnergy);
            ImGui::Text("Quadratic Bending Shell: %e", qbShellEnergy);
        }
    };

    polyscope::show();
}
