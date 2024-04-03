#ifndef SHELLENERGY_H
#define SHELLENERGY_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "../include/MeshConnectivity.h"
#include "../include/RestState.h"
#include "../include/MaterialModel.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/StVKMaterial.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/ElasticShell.h"
#include "QuadraticExpansionBending.h"
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

class ShellEnergy
{
public:
    virtual double elasticEnergy(
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& curEdgeDOFs,        
        bool bendingOnly,
        Eigen::VectorXd* derivative, 
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
        bool bendingOnly,
        Eigen::VectorXd* derivative, 
        std::vector<Eigen::Triplet<double> >* hessian) const
    {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_BENDING;
        if(!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAverageFormulation> mat_;
};

class NeohookeanDirectorShellEnergy : public ShellEnergy
{
public:
    NeohookeanDirectorShellEnergy(
        const LibShell::MeshConnectivity& mesh,
        const LibShell::RestState& restState
    )
        : mesh_(mesh), restState_(restState), mat_() {}

    virtual double elasticEnergy(
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& curEdgeDOFs,
        bool bendingOnly,
        Eigen::VectorXd* derivative,
        std::vector<Eigen::Triplet<double> >* hessian) const
    {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::elasticEnergy(mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAngleTanFormulation> mat_;
};

class StVKShellEnergy : public ShellEnergy
{
  public:
    StVKShellEnergy(
        const LibShell::MeshConnectivity& mesh,
        const LibShell::RestState& restState
        )
        : mesh_(mesh), restState_(restState), mat_() {}

    virtual double elasticEnergy(
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& curEdgeDOFs,
        bool bendingOnly,
        Eigen::VectorXd* derivative,
        std::vector<Eigen::Triplet<double> >* hessian) const
    {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_BENDING;
        if(!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::StVKMaterial<LibShell::MidedgeAverageFormulation> mat_;
};

class StVKDirectorShellEnergy : public ShellEnergy
{
  public:
    StVKDirectorShellEnergy(
        const LibShell::MeshConnectivity& mesh,
        const LibShell::RestState& restState
        )
        : mesh_(mesh), restState_(restState), mat_() {}

    virtual double elasticEnergy(
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& curEdgeDOFs,
        bool bendingOnly,
        Eigen::VectorXd* derivative,
        std::vector<Eigen::Triplet<double> >* hessian) const
    {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::elasticEnergy(mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::StVKMaterial<LibShell::MidedgeAngleTanFormulation> mat_;
};


class QuadraticExpansionShellEnergy : public ShellEnergy
{
public:
    QuadraticExpansionShellEnergy(
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
        bool bendingOnly,
        Eigen::VectorXd* derivative, 
        std::vector<Eigen::Triplet<double> >* hessian) const
    {
        double result = 0;
        int nverts = curPos.rows();
        
        if (derivative)
        {
            derivative->resize(3 * nverts);
            derivative->setZero();
        }
        if(!bendingOnly)
            result += LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(mesh_, curPos, curEdgeDOFs, mat_, restState_, LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_STRETCHING, derivative, hessian);
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

class QuadraticBendingShellEnergy : public ShellEnergy
{
public:
    QuadraticBendingShellEnergy(
        const LibShell::MeshConnectivity& mesh,
        const LibShell::RestState& restState,
        const Eigen::MatrixXd& restPos,
        const Eigen::VectorXd& restEdgeDOFs
    )
        : mesh_(mesh), restState_(restState), mat_(), restPos_(restPos), restEdgeDOFs_(restEdgeDOFs) {
        int nverts = restPos.rows();
        int nfaces = mesh.nFaces();
        int nedges = mesh.nEdges();

        std::vector<Eigen::Triplet<double> > biLcoeffs;
        
        for (int i = 0; i < nedges; i++)
        {
            if (mesh.edgeFace(i, 0) != -1 && mesh.edgeFace(i, 1) != -1)
            {
                int v[4];
                v[0] = mesh.edgeVertex(i, 0);
                v[1] = mesh.edgeVertex(i, 1);
                v[2] = mesh.edgeOppositeVertex(i, 0);
                v[3] = mesh.edgeOppositeVertex(i, 1);
                Eigen::Vector3d e10 = restPos.row(v[0]).transpose() - restPos.row(v[1]).transpose();
                Eigen::Vector3d e12 = restPos.row(v[2]).transpose() - restPos.row(v[1]).transpose();
                Eigen::Vector3d e13 = restPos.row(v[3]).transpose() - restPos.row(v[1]).transpose();
                double c03 = e10.dot(e12) / e10.cross(e12).norm();
                double c04 = e10.dot(e13) / e10.cross(e13).norm();
                Eigen::Vector3d e01 = restPos.row(v[1]).transpose() - restPos.row(v[0]).transpose();
                Eigen::Vector3d e02 = restPos.row(v[2]).transpose() - restPos.row(v[0]).transpose();
                Eigen::Vector3d e03 = restPos.row(v[3]).transpose() - restPos.row(v[0]).transpose();
                double c01 = e01.dot(e02) / e01.cross(e02).norm();
                double c02 = e01.dot(e03) / e01.cross(e03).norm();
                Eigen::Vector4d K(c03 + c04, c01 + c02, -c01 - c03, -c02 - c04);
                double eweight = 0;
                double earea = 0;
                for (int j = 0; j < 2; j++)
                {
                    int face = mesh.edgeFace(i, j);
                    double h = ((LibShell::MonolayerRestState&)restState).thicknesses[face];
                    double lameAlpha = ((LibShell::MonolayerRestState&)restState).lameAlpha[face];
                    double lameBeta = ((LibShell::MonolayerRestState&)restState).lameBeta[face];
                    double weight = h * h * h / 12.0 * (lameAlpha + 2.0 * lameBeta);
                    double area = 0.5 * std::sqrt(((LibShell::MonolayerRestState&)restState).abars[face].determinant());
                    eweight += area * weight;
                    earea += area;
                }

                // each face will be accounted for three times, therefore the division by 3 (making the 3.0 / earea * K * K.transpose() to 1.0 / earea * K * K.transpose())
                Eigen::Matrix4d Q = eweight / earea * 1.0 / earea * K * K.transpose();
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        biLcoeffs.push_back({ v[j], v[k], Q(j,k) });
                    }
                }
            }
        }
        Eigen::SparseMatrix<double> biL(nverts, nverts);
        biL.setFromTriplets(biLcoeffs.begin(), biLcoeffs.end());

        bendingMcoeffs_.clear();

        for (int k = 0; k < biL.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(biL, k); it; ++it)
            {
                for (int j = 0; j < 3; j++)
                {
                    bendingMcoeffs_.push_back({ 3 * (int)it.row() + j, 3 * (int)it.col() + j, it.value() });
                }
            }
        }

        bendingM_.resize(3 * nverts, 3 * nverts);
        bendingM_.setFromTriplets(bendingMcoeffs_.begin(), bendingMcoeffs_.end());
    }

    virtual double elasticEnergy(
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& curEdgeDOFs,
        bool bendingOnly,
        Eigen::VectorXd* derivative, // positions, then thetas
        std::vector<Eigen::Triplet<double> >* hessian) const
    {
        double result = 0;
        int nverts = curPos.rows();

        if (derivative)
        {
            derivative->resize(3 * nverts);
            derivative->setZero();
        }
        
        if(!bendingOnly)
            result += LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(mesh_, curPos, curEdgeDOFs, mat_, restState_, LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_STRETCHING, derivative, hessian);
        
        Eigen::VectorXd displacement(3 * nverts);
        for (int i = 0; i < nverts; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                displacement[3 * i + j] = curPos(i, j);
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