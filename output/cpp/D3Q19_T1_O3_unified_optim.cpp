const double x0 = pow(ux, 2);;
const double x1 = 3*x0;;
const double x2 = pow(uy, 2);;
const double x3 = 3*x2;;
const double x4 = x1 + x3;;
const double x5 = pow(uz, 2);;
const double x6 = 3*x5;;
const double x7 = -tp*ups + ups;;
const double x8 = x6 + x7;;
const double x9 = 2*tp*ups - 2*tp - 2*ups + 2;;
const double x10 = x9 - 2;;
const double x11 = 6*x5;;
const double x12 = -x11;;
const double x13 = 18*uz;;
const double x14 = x13*x2;;
const double x15 = x12 + x14;;
const double x16 = 6*uz;;
const double x17 = -x16;;
const double x18 = x0*x13;;
const double x19 = x17 + x18;;
const double x20 = (1.0/36.0)*rho;;
const double x21 = -x3;;
const double x22 = 2 - x9;;
const double x23 = x21 + x22;;
const double x24 = -x1;;
const double x25 = x11 + x24;;
const double x26 = x14 + x25;;
const double x27 = x10 + x6;;
const double x28 = 6*x2;;
const double x29 = -x28;;
const double x30 = 18*uy;;
const double x31 = x0*x30;;
const double x32 = x29 + x31;;
const double x33 = 6*uy;;
const double x34 = -x33;;
const double x35 = x30*x5;;
const double x36 = x34 + x35;;
const double x37 = -x6;;
const double x38 = x28 + x37;;
const double x39 = x31 + x38;;
const double x40 = 6*x0;;
const double x41 = -x40;;
const double x42 = 18*ux;;
const double x43 = x2*x42;;
const double x44 = x41 + x43;;
const double x45 = 6*ux;;
const double x46 = -x45;;
const double x47 = x42*x5;;
const double x48 = x46 + x47;;
const double x49 = x40 + x43;;
const double x50 = uy*x13;;
const double x51 = x16 + x33 + x35;;
const double x52 = tp*ups + 2*tp - ups;;
const double x53 = x28 + x52;;
const double x54 = x26 + x53;;
const double x55 = (1.0/72.0)*rho;;
const double x56 = -x50;;
const double x57 = -2*tp;;
const double x58 = x57 + x7;;
const double x59 = ux*x13;;
const double x60 = x16 + x18 + x47;;
const double x61 = x45 + x52;;
const double x62 = x11 + x21 + x40 + x61;;
const double x63 = -x59;;
const double x64 = x40 + x46 + x52;;
const double x65 = ux*x30;;
const double x66 = x49 + x61;;
const double x67 = x33 + x39;;
const double x68 = -x65;;
fEq[0] = (1.0/6.0)*rho*(-4*tp - x4 - x8 + 6);
fEq[1] = -x20*(x10 + x15 + x19 + x4);
fEq[2] = x20*(x19 + x23 + x26);
fEq[3] = -x20*(x1 + x27 + x32 + x36);
fEq[4] = x20*(x22 + x24 + x36 + x39);
fEq[5] = -x20*(x27 + x3 + x44 + x48);
fEq[6] = x20*(x23 + x37 + x48 + x49);
fEq[7] = x55*(x50 + x51 + x54);
fEq[8] = x55*(-x14 + x17 + x25 + x33 + x35 + x53 + x56);
fEq[9] = x55*(x16 + x34 - x35 + x54 + x56);
fEq[10] = x55*(-x1 - x15 - x29 - x51 - x56 - x58);
fEq[11] = x55*(x59 + x60 + x62);
fEq[12] = x55*(x17 - x18 + x47 + x62 + x63);
fEq[13] = x55*(x11 + x16 + x18 + x21 - x47 + x63 + x64);
fEq[14] = x55*(-x12 - x3 - x41 - x45 - x58 - x60 - x63);
fEq[15] = x55*(x65 + x66 + x67);
fEq[16] = x55*(-x31 + x34 + x38 + x66 + x68);
fEq[17] = x55*(-x43 + x64 + x67 + x68);
fEq[18] = x55*(-x32 - x33 - x44 - x45 - x57 - x68 - x8);