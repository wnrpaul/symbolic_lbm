fEq[0] = (1.0/3.0)*rho*(-2*tp + 3 - (ux*ux) - (uy*uy) - (uz*uz));
fEq[1] = (1.0/18.0)*rho*(tp + 3*uz - 9*uz*ux*ux - 9*uz*uy*uy - 3*ux*ux - 3*uy*uy + 3*(uz*uz));
fEq[2] = (1.0/18.0)*rho*(tp - 3*uz + 9*uz*(ux*ux) + 9*uz*(uy*uy) - 3*ux*ux - 3*uy*uy + 3*(uz*uz));
fEq[3] = (1.0/18.0)*rho*(tp + 3*uy - 9*uy*ux*ux - 9*uy*uz*uz - 3*ux*ux + 3*(uy*uy) - 3*uz*uz);
fEq[4] = (1.0/18.0)*rho*(tp - 3*uy + 9*uy*(ux*ux) + 9*uy*(uz*uz) - 3*ux*ux + 3*(uy*uy) - 3*uz*uz);
fEq[5] = (1.0/18.0)*rho*(tp + 3*ux - 9*ux*uy*uy - 9*ux*uz*uz + 3*(ux*ux) - 3*uy*uy - 3*uz*uz);
fEq[6] = (1.0/18.0)*rho*(tp - 3*ux + 9*ux*(uy*uy) + 9*ux*(uz*uz) + 3*(ux*ux) - 3*uy*uy - 3*uz*uz);
fEq[7] = (1.0/36.0)*rho*(tp + 9*uy*uz + 3*uy + 9*uy*(uz*uz) + 3*uz + 9*uz*(uy*uy) + 3*(uy*uy) + 3*(uz*uz));
fEq[8] = (1.0/36.0)*rho*(tp - 9*uy*uz + 3*uy + 9*uy*(uz*uz) - 3*uz - 9*uz*uy*uy + 3*(uy*uy) + 3*(uz*uz));
fEq[9] = (1.0/36.0)*rho*(tp - 9*uy*uz - 3*uy - 9*uy*uz*uz + 3*uz + 9*uz*(uy*uy) + 3*(uy*uy) + 3*(uz*uz));
fEq[10] = (1.0/36.0)*rho*(tp + 9*uy*uz - 3*uy - 9*uy*uz*uz - 3*uz - 9*uz*uy*uy + 3*(uy*uy) + 3*(uz*uz));
fEq[11] = (1.0/36.0)*rho*(tp + 9*ux*uz + 3*ux + 9*ux*(uz*uz) + 3*uz + 9*uz*(ux*ux) + 3*(ux*ux) + 3*(uz*uz));
fEq[12] = (1.0/36.0)*rho*(tp - 9*ux*uz + 3*ux + 9*ux*(uz*uz) - 3*uz - 9*uz*ux*ux + 3*(ux*ux) + 3*(uz*uz));
fEq[13] = (1.0/36.0)*rho*(tp - 9*ux*uz - 3*ux - 9*ux*uz*uz + 3*uz + 9*uz*(ux*ux) + 3*(ux*ux) + 3*(uz*uz));
fEq[14] = (1.0/36.0)*rho*(tp + 9*ux*uz - 3*ux - 9*ux*uz*uz - 3*uz - 9*uz*ux*ux + 3*(ux*ux) + 3*(uz*uz));
fEq[15] = (1.0/36.0)*rho*(tp + 9*ux*uy + 3*ux + 9*ux*(uy*uy) + 3*uy + 9*uy*(ux*ux) + 3*(ux*ux) + 3*(uy*uy));
fEq[16] = (1.0/36.0)*rho*(tp - 9*ux*uy + 3*ux + 9*ux*(uy*uy) - 3*uy - 9*uy*ux*ux + 3*(ux*ux) + 3*(uy*uy));
fEq[17] = (1.0/36.0)*rho*(tp - 9*ux*uy - 3*ux - 9*ux*uy*uy + 3*uy + 9*uy*(ux*ux) + 3*(ux*ux) + 3*(uy*uy));
fEq[18] = (1.0/36.0)*rho*(tp + 9*ux*uy - 3*ux - 9*ux*uy*uy - 3*uy - 9*uy*ux*ux + 3*(ux*ux) + 3*(uy*uy));