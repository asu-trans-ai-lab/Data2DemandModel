classdef COD<handle
    properties
        od_id % od的编号
        from_zone_id % od的起始点
        to_zone_id % od的终到点
        target_demand % 目标需求
        estimated_demand % 估计需求
        estimated_theta % logit model 所使用的估计参数
        od_target_gamma % 目标空间分布
        od_estimated_gamma % 目标时间分布
        od_logit_dominator % 该od对应的logit函数的分母 1/exp(gc of path 1+ gc of path 2....)
        od_gamma_numerator_error_signal % 从 od 到 normalized gamma 分子的误差传递梯度
        od_gamma_dominator_error_signal % 从 od 到 normalized gamma 分母的误差传递梯度
        od_path_theta_dominator_error_signal % 当误差从path 传递到theta时，所用的对logit模型分母的求导式（针对od）
        link_od_error_signal % 从link传递到od层的总error
        link_theta_error_signal % 参数theta的调整梯度
        link_gamma_error_signal % 参数gamma的调整梯度
    end 
end

