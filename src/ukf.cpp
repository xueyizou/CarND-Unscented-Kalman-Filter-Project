#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.17;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // State dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // time when the state is true, in us
  time_us_ = 0;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);
  // set weights
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  double weight = 0.5/(n_aug_+lambda_);
  for (int i=1; i<2*n_aug_+1; ++i)
  {
    weights_(i) = weight;
  }

  // the current NIS for radar
  NIS_radar_ = 0;

  // the current NIS for laser
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_)
    {
        Initialize(meas_package);
        time_us_ =  meas_package.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
    }
    else
    {
        if(meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            if(use_laser_)
            {
                double dt = (meas_package.timestamp_ - time_us_)/1000000.0;
                Prediction(dt);
                UpdateLidar(meas_package);
                time_us_= meas_package.timestamp_;
            }

        }
        else
        {
            if(use_radar_)
            {
                double dt = (meas_package.timestamp_ - time_us_)/1000000.0;
                Prediction(dt);
                UpdateRadar(meas_package);
                time_us_= meas_package.timestamp_;
            }

        }

    }
}

/**
 * Initializes Unscented Kalman filter
 */
void UKF::Initialize(MeasurementPackage meas_package)
{
    // first measurement
    std::cout << "Initialization: " << std::endl;

    VectorXd raw_measurements_ = meas_package.raw_measurements_;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        double rho = raw_measurements_(0);
        double phi = raw_measurements_(1);
        double rho_dot = raw_measurements_(2);

        x_ << rho*cos(phi), rho*sin(phi), 0, 0, 0;
        P_ <<1,0,0,0,0,
             0,1,0,0,0,
             0,0,10,0,0,
             0,0,0,10,0,
             0,0,0,0,10;

    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        x_ << raw_measurements_(0), raw_measurements_(1), 0, 0, 0;
        P_ <<1,0,0,0,0,
             0,1,0,0,0,
             0,0,10,0,0,
             0,0,0,10,0,
             0,0,0,0,10;
    }

    std::cout<<"x_= "<<std::endl;
    std::cout<<x_<<std::endl;
    std::cout<<"P_= "<<std::endl;
    std::cout<<P_<<std::endl;
    std::cout<<std::endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    //create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug_; ++i)
    {
        Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }


    //predict sigma points
    for (int i = 0; i< 2*n_aug_+1; ++i)
    {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_pred, py_pred;
        //avoid division by zero
        if (fabs(yawd) > 1e-6)
        {
            px_pred = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_pred = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else
        {
            px_pred = p_x + v*delta_t*cos(yaw);
            py_pred = p_y + v*delta_t*sin(yaw);
        }
        double v_pred = v;
        double yaw_pred = yaw + yawd*delta_t;
        double yawd_pred = yawd;

        //add noise
        px_pred += 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_pred += 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_pred += nu_a*delta_t;
        yaw_pred += 0.5*nu_yawdd*delta_t*delta_t;
        yawd_pred += nu_yawdd*delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_pred;
        Xsig_pred_(1,i) = py_pred;
        Xsig_pred_(2,i) = v_pred;
        Xsig_pred_(3,i) = yaw_pred;
        Xsig_pred_(4,i) = yawd_pred;

    }

    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {//iterate over sigma points
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {//iterate over sigma points
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P_ += weights_(i) * x_diff * x_diff.transpose() ;
    }

    std::cout<<"prediction:"<<std::endl;
    std::cout<<"x_= "<<std::endl;
    std::cout<<x_<<std::endl;
    std::cout<<"P_= "<<std::endl;
    std::cout<<P_<<std::endl;
    std::cout<<std::endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    //set measurement dimension
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_+ 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // measurement model
        Zsig(0,i) = Xsig_pred_(0,i);                        //px
        Zsig(1,i) = Xsig_pred_(1,i);                        //py
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; ++i)
    {
        z_pred += weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S += weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;
    S += R;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    //update state mean and covariance matrix
    x_ = x_+ K * z_diff;
    P_ = P_ - K*S*K.transpose();

    std::cout<<"update_lidar:"<<std::endl;
    std::cout<<"x_= "<<std::endl;
    std::cout<<x_<<std::endl;
    std::cout<<"P_= "<<std::endl;
    std::cout<<P_<<std::endl;
    std::cout<<std::endl;

    NIS_laser_ = z_diff.transpose()*S.inverse()*z_diff;
    std::cout<<"NIS: "<<NIS_laser_<<"  Upper bound: 5.991"<<std::endl;
    std::cout<<std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    //set measurement dimension
    int n_z = 3;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_+ 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double vx = cos(yaw)*v;
        double vy = sin(yaw)*v;

        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
        if(fabs(Zsig(0,i))<1e-6)
        {
            std::cerr<<"division by zero error: rho = "<<Zsig(0,i) <<std::endl;
            return;
        }
        Zsig(1,i) = atan2(p_y,p_x);                                 //phi
        Zsig(2,i) = (p_x*vx + p_y*vy ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; ++i)
    {
        z_pred += weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S += weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0,std_radrd_*std_radrd_;
    S += R;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    //update state mean and covariance matrix
    x_ = x_+ K * z_diff;
    P_ = P_ - K*S*K.transpose();

    std::cout<<"update_radar:"<<std::endl;
    std::cout<<"x_= "<<std::endl;
    std::cout<<x_<<std::endl;
    std::cout<<"P_= "<<std::endl;
    std::cout<<P_<<std::endl;
    std::cout<<std::endl;

    NIS_radar_ = z_diff.transpose()*S.inverse()*z_diff;
    std::cout<<"NIS: "<<NIS_radar_<<"  Upper bound: 7.815"<<std::endl;
    std::cout<<std::endl;
}
