#ifndef QUADRATICBENDING_H
#define QUADRATICBENDING_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include "../include/MeshConnectivity.h"
#include "../include/RestState.h"

template <class SFF>
void bendingMatrix(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::VectorXd& restExtraDOFs,
    const LibShell::RestState& restState,
    double lameAlpha, double lameBeta,
    std::vector<Eigen::Triplet<double> >& Mcoeffs
);


#endif