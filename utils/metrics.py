import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr


class Metrics:
    """
    超分辨率指标计算类
    """
    def __init__(self):
        pass

    def psnr(self, pred, gt, data_range=None):
        """
        峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)

        PSNR = 10 * log10( (data_range^2) / MSE )

        Args:
            data_range: 数据动态范围, 默认取 img2 的有效像素范围
        Returns:
            psnr_value: float
        """
        has_nan = np.isnan(pred).any() or np.isnan(gt).any()
        if has_nan:
            mask = ~(np.isnan(pred) | np.isnan(gt))
            if not np.any(mask):
                return 0.0
            pred_clean = pred[mask]
            gt_clean = gt[mask]
            
            if data_range is None:
                data_range = max(pred_clean.max(), gt_clean.max()) - min(pred_clean.min(), gt_clean.min())
            return psnr(gt_clean, pred_clean, data_range=data_range)
        else:
            if data_range is None:
                data_range = np.max([pred.max(), gt.max()]) - np.min([pred.min(), gt.min()])
            return psnr(gt, pred, data_range=data_range)

    def ssim(self, pred, gt, K1=0.01, K2=0.03, L=None, window_size=11):
        """
        结构相似性指数 (Structural Similarity Index, SSIM)

        SSIM = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) /
               ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))

        其中：
            mu_x, mu_y: 有效像素均值
            sigma_x^2, sigma_y^2: 有效像素方差
            sigma_xy: 有效像素协方差
            C1 = (K1*L)^2, C2 = (K2*L)^2 为稳定常数

        Args:
            K1, K2: 常数，默认 0.01, 0.03
            L: 数据动态范围，如果 None，取有效像素范围
        Returns:
            ssim_val: float
        """
        has_nan = np.isnan(pred).any() or np.isnan(gt).any()
        if has_nan:
            mask = ~(np.isnan(pred) | np.isnan(gt))
            if not np.any(mask):
                return 0.0
            pred_clean = pred[mask]
            gt_clean = gt[mask]
            
            if L is None:
                L = np.max([pred_clean.max(), gt_clean.max()]) - np.min([pred_clean.min(), gt_clean.min()])
        else:
            pred_clean = pred.flatten()
            gt_clean = gt.flatten()
            
            if L is None:
                L = np.max([pred.max(), gt.max()]) - np.min([pred.min(), gt.min()])
        C1 = (K1*L)**2
        C2 = (K2*L)**2
    
        mu_x = np.mean(pred_clean)
        mu_y = np.mean(gt_clean)
        sigma_x2 = np.var(pred_clean)
        sigma_y2 = np.var(gt_clean)
        sigma_xy = np.cov(pred_clean, gt_clean, ddof=0)[0,1]

        ssim_val = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x2 + sigma_y2 + C2))
        return float(ssim_val)

    def mae(self, pred, gt):
        """
        平均绝对误差 (Mean Absolute Error, MAE)

        MAE = mean( | img1 - img2 | ) 只在有效像素计算
        """
        has_nan = np.isnan(pred).any() or np.isnan(gt).any()
        if has_nan:
            mask = ~(np.isnan(pred) | np.isnan(gt))
            if not np.any(mask):
                return 0.0
            return np.mean(np.abs(pred[mask] - gt[mask]))
        else:
            return np.mean(np.abs(pred - gt))

    def mse(self, pred, gt):
        """
        均方误差 (Mean Squared Error, MSE)

        MSE = mean( (img1 - img2)^2 ) 只在有效像素计算
        """
        has_nan = np.isnan(pred).any() or np.isnan(gt).any()
        if has_nan:
            mask = ~(np.isnan(pred) | np.isnan(gt))
            if not np.any(mask):
                return 0.0
            return np.mean((pred[mask] - gt[mask]) ** 2)
        else:
            return np.mean((pred - gt) ** 2)

    def rmse(self, pred, gt):
        """
        均方根误差 (Root Mean Squared Error, RMSE)

        RMSE = sqrt(MSE)
        """
        return np.sqrt(self.mse(pred, gt))

    def relative_l2_error(self, pred, gt):
        """
        相对 L2 误差 (Relative L2 Error)

        Relative L2 = || img1 - img2 ||_2 / || img2 ||_2
        """
        has_nan = np.isnan(pred).any() or np.isnan(gt).any()
        
        if has_nan:
            mask = ~(np.isnan(pred) | np.isnan(gt))
            if not np.any(mask):
                return 0.0
            pred_clean = pred[mask]
            gt_clean = gt[mask]
            
        else:
            pred_clean = pred
            gt_clean = gt
            
        l2_error = np.linalg.norm(pred_clean - gt_clean)
        l2_norm = np.linalg.norm(gt_clean)
        return l2_error / l2_norm if l2_norm > 0 else 0.0

    def __call__(self,pred,gt, *args, **kwds):
        mae = self.mae(pred,gt)
        mse = self.mse(pred,gt)
        rmse = self.rmse(pred,gt)
        relative_l2_error = self.relative_l2_error(pred,gt)
        psnr_value = self.psnr(pred, gt)
        ssim_value = self.ssim(pred, gt)
        return mae,mse,rmse,relative_l2_error,psnr_value,ssim_value