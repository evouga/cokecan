#ifndef SHELLENERGY_H
#define SHELLENERGY_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "../include/MeshConnectivity.h"
#include "../include/RestState.h"
#include "../include/MaterialModel.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/ElasticShell.h"
#include "QuadraticBending.h"

class ShellEnergy
{
public:
    virtual double elasticEnergy(
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& curEdgeDOFs,
        Eigen::VectorXd* derivative, // positions, then thetas
        std::vector<Eigen::Triplet<double> >* hessian) const = 0;
};

class NeohookeanShellEnergy : public ShellEnergy
{
public:
    NeohookeanShellEnergy(
        const LibShell::MeshConnectivity& mesh,
        const LibShell::RestState& restState
    )
        : mesh_(mesh), restState_(restState), mat_() {}

    virtual double elasticEnergy(
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& curEdgeDOFs,
        Eigen::VectorXd* derivative, // positions, then thetas
        std::vector<Eigen::Triplet<double> >* hessian) const
    {
        return LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(mesh_, curPos, curEdgeDOFs, mat_, restState_, derivative, hessian);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAverageFormulation> mat_;
};

class QuadraticBendingShellEnergy : public ShellEnergy
{
public:
    QuadraticBendingShellEnergy(
        const LibShell::MeshConnectivity& mesh,
        const LibShell::RestState& restState,
        const Eigen::MatrixXd &restPos,
        const Eigen::VectorXd &restEdgeDOFs
    )
        : mesh_(mesh), restState_(restState), mat_(), restPos_(restPos), restEdgeDOFs_(restEdgeDOFs) {
        int nverts = restPos.rows();        
        bendingMatrix<LibShell::MidedgeAverageFormulation>(mesh, restPos, restEdgeDOFs, restState, bendingMcoeffs_);
        bendingM_.resize(3 * nverts, 3 * nverts);
        bendingM_.setFromTriplets(bendingMcoeffs_.begin(), bendingMcoeffs_.end());
    }

    virtual double elasticEnergy(
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& curEdgeDOFs,
        Eigen::VectorXd* derivative, // positions, then thetas
        std::vector<Eigen::Triplet<double> >* hessian) const
    {
        double result = LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(mesh_, curPos, curEdgeDOFs, mat_, restState_, LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_STRETCHING, derivative, hessian);        
        int nverts = curPos.rows();
        Eigen::VectorXd displacement(3 * nverts);
        for (int i = 0; i < nverts; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                displacement[3 * i + j] = curPos(i, j) - restPos_(i, j);
            }
        }

        double bendingEnergy = 0.5 * displacement.transpose() * bendingM_ * displacement;
        result += bendingEnergy;

        if (derivative)
        {
            *derivative += bendingM_ * displacement;
        }
        if (hessian)
        {            
            for (auto it : bendingMcoeffs_)
                hessian->push_back(it);
        }

        return result;
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    const Eigen::MatrixXd& restPos_;
    const Eigen::VectorXd& restEdgeDOFs_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAverageFormulation> mat_;
    std::vector<Eigen::Triplet<double> > bendingMcoeffs_;
    Eigen::SparseMatrix<double> bendingM_;
};


#endif