classdef COD<handle
    properties
        od_id % od�ı��
        from_zone_id % od����ʼ��
        to_zone_id % od���յ���
        target_demand % Ŀ������
        estimated_demand % ��������
        estimated_theta % logit model ��ʹ�õĹ��Ʋ���
        od_target_gamma % Ŀ��ռ�ֲ�
        od_estimated_gamma % Ŀ��ʱ��ֲ�
        od_logit_dominator % ��od��Ӧ��logit�����ķ�ĸ 1/exp(gc of path 1+ gc of path 2....)
        od_gamma_numerator_error_signal % �� od �� normalized gamma ���ӵ������ݶ�
        od_gamma_dominator_error_signal % �� od �� normalized gamma ��ĸ�������ݶ�
        od_path_theta_dominator_error_signal % ������path ���ݵ�thetaʱ�����õĶ�logitģ�ͷ�ĸ����ʽ�����od��
        link_od_error_signal % ��link���ݵ�od�����error
        link_theta_error_signal % ����theta�ĵ����ݶ�
        link_gamma_error_signal % ����gamma�ĵ����ݶ�
    end 
end

