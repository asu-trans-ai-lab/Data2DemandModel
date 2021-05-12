function [g_ozone_vector,g_od_vector,g_path_vector,g_link_vector]=Back_propagation(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector,Lambda_sensor)
[~,nb_ozone]=size(g_ozone_vector);
[~,nb_od]=size(g_od_vector);
[~,nb_path]=size(g_path_vector);
[~,nb_link]=size(g_link_vector);
%================from link to path===============         
        for p=1:nb_path
            g_path_vector(p).link_path_error_signal=0;
            for li=1:nb_link
                if size(str2num(g_link_vector(li).count_sensor_name{1}),1)~=1 
                    if g_path_vector(p).pass_link_vector(g_link_vector(li).link_id)==1
                        g_path_vector(p).link_path_error_signal=g_path_vector(p).link_path_error_signal...
                            +Lambda_sensor*(g_link_vector(li).estimated_flow/g_link_vector(li).target_flow-1)*(1/g_link_vector(li).target_flow);
                    end
                end
            end
        end        
fprintf('Back propagate the error signals from the layer of links to the layer of paths\n')        
for od=1:nb_od
    g_od_vector(od).link_theta_error_signal=0;
    for p=1:nb_path
        if g_path_vector(p).from_zone_id==g_od_vector(od).from_zone_id...
                && g_path_vector(p).to_zone_id==g_od_vector(od).to_zone_id            
            g_od_vector(od).link_theta_error_signal=g_od_vector(od).link_theta_error_signal...
                +g_path_vector(p).link_path_error_signal...
                *(g_path_vector(p).path_theta_numerator_error_signal+g_path_vector(p).path_theta_dominator_error_signal);           
        end
    end
end
fprintf('Back propagate the error signals from the layer of paths to the parameter theta\n')   

for od=1:nb_od
    g_od_vector(od).link_od_error_signal=0;
    for p=1:nb_path
        if g_path_vector(p).from_zone_id==g_od_vector(od).from_zone_id...
                && g_path_vector(p).to_zone_id==g_od_vector(od).to_zone_id
            g_od_vector(od).link_od_error_signal=g_od_vector(od).link_od_error_signal...
                +(g_path_vector(p).link_path_error_signal*g_path_vector(p).path_estimated_proportion);           
        end
    end
end
fprintf('Back propagate the error signals from the layer of paths to the layer of od\n')

for od=1:nb_od
    g_od_vector(od).link_gamma_error_signal=0;
    for o=1:nb_ozone
        if g_od_vector(od).from_zone_id==g_ozone_vector(o).ozone_id
% 原来的代码为g_od_vector(od).link_gamma_error_signal=g_od_vector(od).link_od_error_signal*g_ozone_vector(o).estimated_generation;
g_od_vector(od).link_gamma_error_signal=g_od_vector(od).link_od_error_signal*(g_od_vector(od).od_gamma_numerator_error_signal+...
    g_od_vector(od).od_gamma_dominator_error_signal);

        end
    end
end
fprintf('Back propagate the error signals from the layer of od to the parameter gamma\n')   

for o=1:nb_ozone
    g_ozone_vector(o).link_ozone_error_signal=0;
    for od=1:nb_od
        if g_od_vector(od).from_zone_id==g_ozone_vector(o).ozone_id
            g_ozone_vector(o).link_ozone_error_signal...
                =g_ozone_vector(o).link_ozone_error_signal...
                +(g_od_vector(od).link_od_error_signal*g_od_vector(od).od_estimated_gamma);
        end
    end
end
fprintf('Back propagate the error signals from the layer of od to the layer of ozone\n') 
end