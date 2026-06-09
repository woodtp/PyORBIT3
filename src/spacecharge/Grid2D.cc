//This class repersents a 2D rectangular grid

#include "Grid2D.hh"
#include "ParticleMacroSize.hh"
// #include "BufferStore.hh"

#include <numeric>
#include <iostream>

using namespace OrbitUtils;

// Constructor
Grid2D::Grid2D(int xSize, int ySize): CppPyWrapper(NULL),
	data_(xSize * ySize),
	xSize_(xSize),
	ySize_(ySize),
	xMin_(-1.0),
	xMax_(+1.0),
	yMin_(-1.0),
	yMax_(+1.0)
{
	init();
	// setZero();
}

Grid2D::Grid2D(int xSize, int ySize,
	       double xMin, double xMax,
	       double yMin, double yMax): CppPyWrapper(NULL),
	data_(xSize * ySize),
	xSize_(xSize),
	ySize_(ySize),
	xMin_(xMin),
	xMax_(xMax),
	yMin_(yMin),
	yMax_(yMax)
{
	init();
	// setZero();
}

Grid2D::Grid2D(double* externalData,
		size_t xSize, size_t ySize,
		double xMin, double xMax,
		double yMin, double yMax): CppPyWrapper(NULL),
	owns_(false), externalData_(externalData),
	xSize_(xSize), ySize_(ySize),
	xMin_(xMin), xMax_(xMax),
	yMin_(yMin), yMax_(yMax)
{
  init();
  // setZero();
}

void Grid2D::init(){
  if(xSize_ < 3 || ySize_ < 3) {
    int rank = 0;
    ORBIT_MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cerr << "Grid2D::Grid2D - CONSTRUCTOR \n"
                << "The grid size too small (should be more than 3)! \n"
                << "number x bins ="<< xSize_ <<" \n"
                << "number y bins ="<< ySize_ <<" \n"
                << "Stop. \n";
    }
    ORBIT_MPI_Finalize();
  }

	dx_ = (xMax_ - xMin_)/(xSize_ -1);
	dy_ = (yMax_ - yMin_)/(ySize_ -1);


	double *ptr = owns_ ? data_.data() : externalData_;

	rows_.resize(xSize_);
	for (size_t ix = 0; ix < xSize_; ++ix) {
	  rows_[ix] = ptr + ix*ySize_;
	}

	arr_ = rows_.data();
}

// Destructor
// Grid2D::~Grid2D()
// {
	//std::cerr<<"debug Grid2D::~Grid2D()"<<std::endl;
	// for(int i = 0; i < xSize_; i++){
	// 	delete [] arr_[i];
	// }
	// delete [] arr_;
// }

/** Sets the value to the one point of the 2D grid  */
void Grid2D::setValue(double value, int ix, int iy){
	arr_[ix][iy] = value;
}

/** Returns the value on grid*/
double Grid2D::getValueOnGrid(int ix, int iy){
	return arr_[ix][iy];
}

double& Grid2D::operator()(size_t ix, size_t iy) {
  return at(ix, iy);
}

const double& Grid2D::operator()(size_t ix, size_t iy) const {
  return at(ix, iy);
}

double& Grid2D::at(size_t ix, size_t iy) {
  if (ix >= xSize_ || iy >= ySize_) throw std::out_of_range("Grid2D::at");
  return elem(ix, iy);
}

const double& Grid2D::at(size_t ix, size_t iy) const {
  if (ix >= xSize_ || iy >= ySize_) throw std::out_of_range("Grid2D::at");
  return elem(ix, iy);
}

double& Grid2D::elem(size_t ix, size_t iy) {
  return data_[ix*ySize_ + iy];
}

const double& Grid2D::elem(size_t ix, size_t iy) const {
  return data_[ix*ySize_ + iy];
}

/** Returns the interpolated value from the 2D grid */
double Grid2D::getValue(double x, double y){
	int iX, iY;
	double Wxm, Wx0, Wxp, Wym, Wy0, Wyp;
	double xFract,  yFract;
	double xFract2, yFract2;
	getIndAndFracX(x,iX,xFract);
	getIndAndFracY(y,iY,yFract);
	xFract2 = xFract * xFract;
	yFract2 = yFract * yFract;
	Wxm = 0.5 * (0.25 - xFract + xFract2);
	Wx0 = 0.75 - xFract2;
	Wxp = 0.5 * (0.25 + xFract + xFract2);
	Wym = 0.5 * (0.25 - yFract + yFract2);
	Wy0 = 0.75 - yFract2;
	Wyp = 0.5 * (0.25 + yFract + yFract2);
	double value =
	Wxm * Wym * elem(iX-1, iY-1) +
	Wxm * Wy0 * elem(iX-1, iY)   +
	Wxm * Wyp * elem(iX-1, iY+1) +
	Wx0 * Wym * elem(iX,   iY-1) +
	Wx0 * Wy0 * elem(iX,   iY)   +
	Wx0 * Wyp * elem(iX,   iY+1) +
	Wxp * Wym * elem(iX+1, iY-1) +
	Wxp * Wy0 * elem(iX+1, iY)   +
	Wxp * Wyp * elem(iX+1, iY+1);
	return value;
}

/** Bins the Bunch into the 2D grid using X and Y coordinates.
    If bunch has a macrosize particle attribute it will be used.
  */
void Grid2D::binBunch(Bunch* bunch){
	this->binBunch(bunch,0,2);
}

/** Bins the Bunch into the 2D grid using coordinate indexes ind0 and ind1.
    If bunch has a macrosize particle attribute it will be used.
  */
void Grid2D::binBunch(Bunch* bunch, int ind0, int ind1){
	bunch->compress();
	double** part_coord_arr = bunch->coordArr();
	int has_msize = bunch->hasParticleAttributes("macrosize");
	if(has_msize > 0){
		ParticleMacroSize* macroSizeAttr = (ParticleMacroSize*) bunch->getParticleAttributes("macrosize");
		double m_size = 0.;
		for(int i = 0, n = bunch->getSize(); i < n; i++){
			m_size = macroSizeAttr->macrosize(i);
			binValue(m_size,part_coord_arr[i][ind0],part_coord_arr[i][ind1]);
		}
		return;
	}
	double m_size = bunch->getMacroSize();
	int nParts = bunch->getSize();
	for(int i = 0; i < nParts; i++){
		binValue(m_size,part_coord_arr[i][ind0],part_coord_arr[i][ind1]);
	}
}

/** Bins the value into the 2D grid */
void Grid2D::binValue(double value, double x, double y){
	if(x < xMin_ || x > xMax_ || y < yMin_ || y > yMax_) return;

	int iX, iY;
	double xFract,  yFract;
	getIndAndFracX(x, iX, xFract);
	getIndAndFracY(y, iY, yFract);

	const double xFract2 = xFract * xFract;
	const double yFract2 = yFract * yFract;

	const double Wxm = 0.5 * (0.25 - xFract + xFract2);
	const double Wx0 = 0.75 - xFract2;
	const double Wxp = 0.5 * (0.25 + xFract + xFract2);
	const double Wym = 0.5 * (0.25 - yFract + yFract2);
	const double Wy0 = 0.75 - yFract2;
	const double Wyp = 0.5 * (0.25 + yFract + yFract2);

	elem(iX-1, iY-1) += Wxm * Wym * value;
        elem(iX-1, iY  ) += Wxm * Wy0 * value;
        elem(iX-1, iY+1) += Wxm * Wyp * value;
        elem(iX,   iY-1) += Wx0 * Wym * value;
        elem(iX,   iY  ) += Wx0 * Wy0 * value;
        elem(iX,   iY+1) += Wx0 * Wyp * value;
        elem(iX+1, iY-1) += Wxp * Wym * value;
        elem(iX+1, iY  ) += Wxp * Wy0 * value;
        elem(iX+1, iY+1) += Wxp * Wyp * value;
}

/** Does a bilinear binning scheme on the bunch using X and Y coordinates */
void Grid2D::binBunchBilinear(Bunch* bunch){
	this->binBunchBilinear(bunch,0,2);
}

/** Does a bilinear binning scheme on the bunch using coordinate indexes ind0 and ind1.
  */
void Grid2D::binBunchBilinear(Bunch* bunch, int ind0, int ind1){
	bunch->compress();
	double** part_coord_arr = bunch->coordArr();
	int nParts = bunch->getSize();
	int has_msize = bunch->hasParticleAttributes("macrosize");
	if(has_msize > 0){
		ParticleMacroSize* macroSizeAttr = (ParticleMacroSize*) bunch->getParticleAttributes("macrosize");
			double m_size = 0.;
			for(int i = 0, n = bunch->getSize(); i < n; i++){
				m_size = macroSizeAttr->macrosize(i);
				binValueBilinear(m_size,part_coord_arr[i][ind0],part_coord_arr[i][ind1]);
			}
			return;
		}
		double m_size = bunch->getMacroSize();
		for(int i = 0; i < nParts; i++){
			//cerr<<"i = "<<i;
			binValueBilinear(m_size,part_coord_arr[i][ind0],part_coord_arr[i][ind1]);
		}
}

/** Bilinear bin of the value into the 2D grid */
void Grid2D::binValueBilinear(double value, double x, double y){

	int iX, iY;
	double xFract,  yFract;

	getBilinearIndAndFracX(x, iX, xFract);
	getBilinearIndAndFracY(y, iY, yFract);

	elem(iX,   iY  ) += ((1.-xFract) * (1.-yFract)) * value;
	elem(iX,   iY+1) += ((1.-xFract) * yFract) * value;
	elem(iX+1, iY  ) += (xFract * (1.-yFract)) * value;
	elem(iX+1, iY+1) += (xFract * yFract) * value;
}


/** Calculates gradient at a position (x,y) by using 9-points schema */
void Grid2D::calcGradient(double x, double y, double& ex, double& ey){
	int iX, iY;
	double xFract,  yFract;

	getIndAndFracX(x,iX,xFract);
	getIndAndFracY(y,iY,yFract);

	const double xFract2 = xFract * xFract;
	const double yFract2 = yFract * yFract;

	const double Wxm = 0.5 * (0.25 - xFract + xFract2);
	const double Wx0 = 0.75 - xFract2;
	const double Wxp = 0.5 * (0.25 + xFract + xFract2);
	const double Wym = 0.5 * (0.25 - yFract + yFract2);
	const double Wy0 = 0.75 - yFract2;
	const double Wyp = 0.5 * (0.25 + yFract + yFract2);

        const double dWxm = (-1.0)*(0.5 - xFract);
	const double dWx0 = (-1.0)*(+2.) * xFract;
	const double dWxp = (-1.0)*(-(0.5 + xFract));
	const double dWym = (-1.0)*(0.5 - yFract);
	const double dWy0 = (-1.0)*(+2.) * yFract;
	const double dWyp = (-1.0)*(-(0.5 + yFract));
  ex =
	dWxm * Wym * elem(iX-1, iY-1) +
	dWxm * Wy0 * elem(iX-1, iY  ) +
	dWxm * Wyp * elem(iX-1, iY+1) +
	dWx0 * Wym * elem(iX,   iY-1) +
	dWx0 * Wy0 * elem(iX,   iY  ) +
	dWx0 * Wyp * elem(iX,   iY+1) +
	dWxp * Wym * elem(iX+1, iY-1) +
	dWxp * Wy0 * elem(iX+1, iY  ) +
	dWxp * Wyp * elem(iX+1, iY+1);
	ex = ex / dx_;
  ey =
	Wxm * dWym * elem(iX-1, iY-1) +
	Wxm * dWy0 * elem(iX-1, iY  ) +
	Wxm * dWyp * elem(iX-1, iY+1) +
	Wx0 * dWym * elem(iX,   iY-1) +
	Wx0 * dWy0 * elem(iX,   iY  ) +
	Wx0 * dWyp * elem(iX,   iY+1) +
	Wxp * dWym * elem(iX+1, iY-1) +
	Wxp * dWy0 * elem(iX+1, iY  ) +
	Wxp * dWyp * elem(iX+1, iY+1);
	ey = ey / dy_;
}

/** Calculates gradient at a grid point (ix,iy) by using 9-points schema */
void Grid2D::calcGradient(int iX, int iY, double& ex, double& ey){
	if(iX != 0 && iX != (xSize_ - 1) && iY != 0 && iY != (ySize_ - 1)){
		ex =
		(- 0.125) * arr_[iX-1][iY-1] +
		(- 0.75 )* arr_[iX-1][iY]   +
		(- 0.125) * arr_[iX-1][iY+1] +
		0.125 * arr_[iX+1][iY-1] +
		0.75  * arr_[iX+1][iY]   +
		0.125 * arr_[iX+1][iY+1];
		ex = 0.5 * ex / dx_;
		ey =
		(-0.125) * arr_[iX-1][iY-1] +
		( 0.125) * arr_[iX-1][iY+1] +
		(-0.75 ) * arr_[iX]  [iY-1] +
		( 0.75 ) * arr_[iX]  [iY+1] +
		(-0.125) * arr_[iX+1][iY-1] +
		( 0.125) * arr_[iX+1][iY+1];
		ey = 0.5 * ey / dy_;
		return;
	}
	double x = xMin_ + iX*dx_;
	double y = yMin_ + iY*dy_;
	calcGradient(x,y,ex,ey);
}

/** Calculates bilinear interpolated gradient at a position (x,y)*/
void Grid2D::calcGradientBilinear(double x, double y, double& ex, double& ey){
	int iX, iY;
	double xFract,  yFract;

	getBilinearIndAndFracX(x,iX,xFract);
	getBilinearIndAndFracY(y,iY,yFract);

  ex = (elem(iX+1, iY) - elem(iX, iY))*(1.0 - yFract) + yFract*(elem(iX+1, iY+1) - elem(iX, iY+1));
  ex = ex / dx_;

  ey =(elem(iX, iY+1) - elem(iX, iY))*(1.0 - xFract) + xFract*(elem(iX+1, iY+1) - elem(iX+1, iY));
  ey = ey / dy_;
}

/** Calculates bilinear interpolated value at a position (x,y) */
void Grid2D::interpolateBilinear(double x, double y, double& value){
	int iX, iY;
	double xFract,  yFract;

	getBilinearIndAndFracX(x,iX,xFract);
	getBilinearIndAndFracY(y,iY,yFract);

	const double f1 = elem(iX  , iY  );
	const double f2 = elem(iX+1, iY  );
	const double f3 = elem(iX+1, iY+1);
	const double f4 = elem(iX  , iY+1);

	value = (1. - xFract) * (1. - yFract) * f1 + xFract * (1. - yFract) * f2 +
	(1. - xFract) * yFract * f4 + xFract * yFract * f3;

}
/** Sets all grid points to zero */
void Grid2D::setZero() {
  double* data_ptr = owns_ ? data_.data() : externalData_;
  std::fill(data_ptr, data_ptr + xSize_ * ySize_, 0.0);
}

/** Returns the reference to the 2D array */
double** Grid2D::getArr(){
	return arr_;
}

/** Returns the grid size in x-direction */
int Grid2D::getSizeX(){
	return xSize_;
}

/** Returns the grid size in y-direction */
int Grid2D::getSizeY(){
	return ySize_;
}

void Grid2D::getIndAndFracX(double x, int& ind, double& frac){
   ind  = int ( (x - xMin_)/dx_ + 0.5 );
   if(ind < 1) ind = 1;
   if(ind > (xSize_-2)) ind = xSize_ - 2;
   frac = (x - (xMin_ + ind*dx_))/dx_;
}

void Grid2D::getIndAndFracY(double y, int& ind, double& frac){
   ind  = int ( (y - yMin_)/dy_ + 0.5 );
   if(ind < 1) ind = 1;
   if(ind > (ySize_-2)) ind = ySize_ - 2;
   frac = (y - (yMin_ + ind*dy_))/dy_;
}

/** Returns the index and fraction for a bilinear scheme */
void Grid2D::getBilinearIndAndFracX(double x, int& ind, double& frac){
	ind  = int ( (x - xMin_)/dx_ );
	if(ind < 0) ind = 0;
	if(ind > (xSize_-2)) ind = xSize_ - 2;
	frac = (x - (xMin_ + ind*dx_))/dx_;
}

/** Returns the index and fraction for a bilinear scheme */
void Grid2D::getBilinearIndAndFracY(double y, int& ind, double& frac){
	ind  = int ( (y - yMin_)/dy_ );
	if(ind < 0) ind = 0;
	if(ind > (ySize_-2)) ind = ySize_ - 2;
	frac = (y - (yMin_ + ind*dy_))/dy_;

}

/** Returns the grid point x-coordinate for this index. */
double Grid2D::getGridX(int index){
	return xMin_ + index*dx_;
}

/** Returns the grid point y-coordinate for this index. */
double Grid2D::getGridY(int index){
	return yMin_ + index*dy_;
}

/** Returns the grid step along x-axis */
double Grid2D::getStepX(){
	return dx_;
}

/** Returns the grid step along y-axis */
double Grid2D::getStepY(){
	return dy_;
}

/** Returns the max x in the grid points */
double Grid2D::getMaxX(){return xMax_;};

/** Returns the min x in the grid points */
double Grid2D::getMinX(){return xMin_;};

/** Returns the max y in the grid points */
double Grid2D::getMaxY(){return yMax_;};

/** Returns the min y in the grid points */
double Grid2D::getMinY(){return yMin_;};

/** Returns the sum of all grid points */
double Grid2D::getSum(){
  double* data_ptr = owns_ ? data_.data() : externalData_;
  return std::accumulate(data_ptr, data_ptr + xSize_ * ySize_, 0.0);
};


/** Multiply all elements of Grid2D by constant coefficient */
void Grid2D::multiply(double coeff){
  double* data_ptr = owns_ ? data_.data() : externalData_;
  for (size_t i = 0; i < xSize_ * ySize_; ++i) {
    data_ptr[i] *= coeff;
  }
}

/** Sets x-grid */
void Grid2D::setGridX(double xMin, double xMax){
	xMin_ = xMin;
	xMax_ = xMax;
	dx_ = (xMax_ - xMin_)/(xSize_ -1);
}

/** Sets y-grid */
void Grid2D::setGridY(double yMin, double yMax){
	yMin_ = yMin;
	yMax_ = yMax;
	dy_ = (yMax_ - yMin_)/(ySize_ -1);
}

/** Returns 1 if (x,y) is inside the grid region, and 0 otherwise */
int Grid2D::isInside(double x,double y){
	if(x < xMin_ || x > xMax_) return 0;
	if(y < yMin_ || y > yMax_) return 0;
	return 1;
}

/**synchronizeMPI */
void Grid2D::synchronizeMPI(pyORBIT_MPI_Comm* pyComm){
#if USE_MPI > 0
  // ====== MPI  start ========
    const int n = xSize_ * ySize_;
    double* ptr = owns_ ? data_.data() : externalData_;
    MPI_Comm comm = (pyComm == nullptr) ? MPI_COMM_WORLD : pyComm->comm;

    ORBIT_MPI_Allreduce(MPI_IN_PLACE, ptr, n, MPI_DOUBLE, MPI_SUM, comm);
  // ===== MPI end =====
#else // no-op
        (void)pyComm;
#endif // USE_MPI > 0
}
