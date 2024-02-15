#ifndef STATICSOLVE_H
#define STATICSOLVE_H

#include <Eigen/Core>
#include <vector>
#include <set>

#include "../include/MaterialModel.h"
#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "ShellEnergy.h"

void takeOneStep(const ShellEnergy &energyModel,
    Eigen::MatrixXd& curPos,
    Eigen::VectorXd& curEdgeDOFs,
    const std::set<int> &fixed,
    double tol,
    double& reg)
{

    int nverts = (int)curPos.rows();
    int nedgedofs = curEdgeDOFs.size();
    int nfixed = fixed.size();

    int fullDOFs = 3 * nverts + nedgedofs;
    int freeDOFs = 3 * (nverts - nfixed) + nedgedofs;

    std::vector<Eigen::Triplet<double> > Pcoeffs;
    int row = 0;
    for (int i = 0; i < nverts; i++)
    {
        if (!fixed.count(i))
        {
            for (int j = 0; j < 3; j++)
            {
                Pcoeffs.push_back({ row, 3 * i + j, 1.0 });
                row++;
            }
        }
    }
    for (int i = 0; i < nedgedofs; i++)
    {
        Pcoeffs.push_back({ row, 3 * nverts + i, 1.0 });
        row++;
    }
    Eigen::SparseMatrix<double> P(freeDOFs, fullDOFs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());

    while (true)
    {
        Eigen::VectorXd derivative;
        std::vector<Eigen::Triplet<double> > hessian;

        double energy = energyModel.elasticEnergy(curPos, curEdgeDOFs, false, &derivative, &hessian);

        Eigen::SparseMatrix<double> H(fullDOFs, fullDOFs);
        H.setFromTriplets(hessian.begin(), hessian.end());

        Eigen::VectorXd force = -P*derivative;
        Eigen::SparseMatrix<double> I(freeDOFs, freeDOFs);
        I.setIdentity();
        Eigen::SparseMatrix<double> projH = P * H * P.transpose();
        projH += reg * I;
        
        Eigen::VectorXd maxvals(freeDOFs);
        maxvals.setZero();
        for (int k = 0; k < projH.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(projH, k); it; ++it)
            {
                maxvals[it.row()] = std::max(maxvals[it.row()], std::fabs(it.value()));
            }
        }
        std::vector<Eigen::Triplet<double> > Dcoeffs;
        for (int i = 0; i < freeDOFs; i++)
        {
            double val = (maxvals[i] == 0.0 ? 1.0 : 1.0 / std::sqrt(maxvals[i]));
            Dcoeffs.push_back({ i,i, val });
        }
        Eigen::SparseMatrix<double> D(freeDOFs, freeDOFs);
        D.setFromTriplets(Dcoeffs.begin(), Dcoeffs.end());

        Eigen::SparseMatrix<double> DHDT = D * projH * D.transpose();

        std::cout << "solving, original force residual: " << force.norm() << std::endl;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(DHDT);
        if (solver.info() == Eigen::Success)
        {
            Eigen::VectorXd rhs = D * force;
            Eigen::VectorXd descentDir = P.transpose() * D * solver.solve(rhs);
            std::cout << "solved" << std::endl;
            Eigen::MatrixXd newPos = curPos;
            for (int i = 0; i < nverts; i++)
            {
                newPos.row(i) += descentDir.segment<3>(3 * i);
            }
            Eigen::VectorXd newEdgeDofs = curEdgeDOFs + descentDir.segment(3 * nverts, nedgedofs);



            double newenergy = energyModel.elasticEnergy(newPos, newEdgeDofs, false, &derivative, NULL);
            force = -P * derivative;

            double forceResidual = force.norm();

            if (newenergy <= energy)
            {
                std::cout << "Old energy: " << energy << " new energy: " << newenergy << " force residual " << forceResidual << " pos change " << descentDir.segment(0, 3 * nverts).norm() << " theta change " << descentDir.segment(3 * nverts, nedgedofs).norm() << " lambda " << reg << std::endl;
                curPos = newPos;
                curEdgeDOFs = newEdgeDofs;
                reg /= 2.0;
                if(forceResidual < tol)
                    break;
                continue;
            }
            else
            {
                std::cout << "Not a descent direction; old energy: " << energy << " new energy: " << newenergy << " lambda now: " << 2.0 * reg << std::endl;
            }
        }
        else
        {
            std::cout << "Matrix not positive-definite, lambda now " << reg << std::endl;
        }

        reg *= 2.0;

    }
}

#endif