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

double QBEnergy;

void lameParameters(double& alpha, double& beta)
{
    double young = 1.0 / thickness; // doesn't matter for static solves
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

struct Energies
{
    double exact;
    double quadraticbending;
    double neohookean;
};

Energies measureEnergy(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::MatrixXd& curPos,
    double thickness,
    double lameAlpha,
    double lameBeta,
    double curRadius,
    double curHeight,
    Eigen::MatrixXd &nhForces,
    Eigen::MatrixXd &qbForces)
{
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edgeDOFs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edgeDOFs, mesh, restPos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState restState;

    // set uniform thicknesses
    restState.thicknesses.resize(mesh.nFaces(), thickness);
    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::firstFundamentalForms(mesh, restPos, restState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, restPos, edgeDOFs, restState.bbars);

    // Make the half-cylinder rest-flat
    for (int i = 0; i < mesh.nFaces(); i++)
        restState.bbars[i].setZero();

    Eigen::VectorXd restEdgeDOFs = edgeDOFs;

    NeohookeanShellEnergy nhenergyModel(mesh, restState);
    QuadraticBendingShellEnergy qbenergyModel(mesh, restState, restPos, restEdgeDOFs);

    Eigen::VectorXd nhF;
    Eigen::VectorXd qbF;
    result.neohookean = nhenergyModel.elasticEnergy(curPos, edgeDOFs, true, &nhF, NULL);
    result.quadraticbending = qbenergyModel.elasticEnergy(curPos, edgeDOFs, true, &qbF, NULL);

    int nverts = curPos.rows();
    nhForces.resize(nverts, 3);
    qbForces.resize(nverts, 3);
    for (int i = 0; i < nverts; i++)
    {
        nhForces.row(i) = -nhF.segment<3>(3 * i);
        qbForces.row(i) = -qbF.segment<3>(3 * i);
    }

    // ground truth energy
    // W = PI * r    
    // r(x,y) = (r cos[x/r], r sin[x/r], y)^T
    // dr(x,y) = ((-sin[x/r], 0),
    //            ( cos[x/r], 0),
    //            ( 0, 1 ))
    Eigen::Matrix2d abar;
    abar.setIdentity();

    // n = (-sin[x/r], cos[x/r], 0) x (0, 0, 1) = ( cos[x/r], sin[x/r], 0 )
    // dn = ((-sin[x/r]/r, 0),
    //       ( cos[x/r]/r, 0),
    //       ( 0, 0 ))
    // b = dr^T dn = ((1/r, 0), (0, 0))
    Eigen::Matrix2d b;
    b << 1.0 / curRadius, 0, 0, 0;

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm = lameAlpha / 2.0 * M.trace() * M.trace() + lameBeta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 24.0;
    constexpr double PI = 3.1415926535898;
    double area = PI * curRadius * curHeight;

    result.exact = svnorm * coeff * area;


    return result;
}

int main(int argc, char* argv[])
{
    cokeRadius = 0.0325;
    cokeHeight = 0.122;

    triangleArea = 0.000001;

    Energies curenergies;
    QBEnergy = 0;

    Eigen::MatrixXd nhForces;
    Eigen::MatrixXd qbForces;


    // set up material parameters
    thickness = 0.00010;
    poisson = 1.0 / 2.0;

    // load mesh

    Eigen::MatrixXd origV;
    Eigen::MatrixXd rolledV;
    Eigen::MatrixXi F;

    makeHalfCylinder(cokeRadius, cokeHeight, triangleArea, origV, rolledV, F);
    double lameAlpha, lameBeta;
    lameParameters(lameAlpha, lameBeta);
    LibShell::MeshConnectivity mesh(F);
    double curRadius = cokeRadius;
    double curHeight = cokeHeight;

    curenergies = measureEnergy(mesh, origV, rolledV, thickness, lameAlpha, lameBeta, curRadius, curHeight, nhForces, qbForces);
    
    polyscope::init();
    polyscope::registerSurfaceMesh("Input Mesh", origV, F);
    auto *surf = polyscope::registerSurfaceMesh("Rolled Mesh", rolledV, F);

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
                makeHalfCylinder(cokeRadius, cokeHeight, triangleArea, origV, rolledV, F);   
                polyscope::registerSurfaceMesh("Input Mesh", origV, F);
                surf = polyscope::registerSurfaceMesh("Rolled Mesh", rolledV, F);
                curRadius = cokeRadius;
                curHeight = cokeHeight;
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
            curenergies = measureEnergy(mesh, origV, rolledV, thickness, lameAlpha, lameBeta, curRadius, curHeight, nhForces, qbForces);
            surf->addVertexVectorQuantity("NH Force", nhForces);
            surf->addVertexVectorQuantity("QB Force", qbForces);
        }        

        if (ImGui::CollapsingHeader("Energies", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Neohookean Shell: %e", curenergies.neohookean);
            ImGui::Text("Quadratic Bending Shell: %e", curenergies.quadraticbending);
            ImGui::Text("Exact Shell: %e", curenergies.exact);
        }
    };

    polyscope::show();
}
