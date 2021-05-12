function [g_ozone_vector,g_od_vector,g_path_vector,g_link_vector]=Preliminary_derivatives(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector)
[~,nb_ozone]=size(g_ozone_vector);
[~,nb_od]=size(g_od_vector);
[~,nb_path]=size(g_path_vector);
[~,nb_link]=size(g_link_vector);

for od=1:nb_od
    % calculate the deriviative of the dominator of logit model 
    g_od_vector(od).od_path_theta_dominator_error_signal=0;
    for p=1:nb_path
        if g_path_vector(p).from_zone_id==g_od_vector(od).from_zone_id && g_path_vector(p).to_zone_id==g_od_vector(od).to_zone_id
            g_od_vector(od).od_path_theta_dominator_error_signal...
                = g_od_vector(od).od_path_theta_dominator_error_signal...
                + (-g_path_vector(p).path_travel_time)...
                * exp(-g_od_vector(od).estimated_theta*g_path_vector(p).path_travel_time-g_path_vector(p).path_cost);
        end
    end
%原来的代码为 g_od_vector(od).od_path_theta_dominator_error_signal=-(1/g_od_vector(od).od_path_theta_dominator_error_signal^2);
g_od_vector(od).od_path_theta_dominator_error_signal=-(1/g_od_vector(od).od_logit_dominator^2)*g_od_vector(od).od_path_theta_dominator_error_signal;


end

for p=1:nb_path
    for od=1:nb_od
        if g_path_vector(p).from_zone_id==g_od_vector(od).from_zone_id && g_path_vector(p).to_zone_id==g_od_vector(od).to_zone_id
            g_path_vector(p).path_theta_dominator_error_signal...
                =g_od_vector(od).estimated_demand...
                *exp(-g_path_vector(p).path_travel_time*g_od_vector(od).estimated_theta-g_path_vector(p).path_cost)...
                *g_od_vector(od).od_path_theta_dominator_error_signal;
        end
    end
end
fprintf('Calculate the deriviative of the dominator of logit model from the layer of paths to the parameter theta \n')

for p=1:nb_path
    for od=1:nb_od
        if g_path_vector(p).from_zone_id==g_od_vector(od).from_zone_id && g_path_vector(p).to_zone_id==g_od_vector(od).to_zone_id
            g_path_vector(p).path_theta_numerator_error_signal...
                =g_od_vector(od).estimated_demand...
                *g_od_vector(od).od_logit_dominator...
                *(-g_path_vector(p).path_travel_time)...
                *exp(-g_od_vector(od).estimated_theta*g_path_vector(p).path_travel_time-g_path_vector(p).path_cost);
            %g_od_vector(od).estimated_demand
            %g_path_vector(p).path_theta_numerator_error_signal
        end
    end
end

fprintf('Calculate the deriviative of the numerator of logit model from the layer of paths to the parameter theta\n')
    
for od=1:nb_od
    for o=1:nb_ozone
        if g_ozone_vector(o).ozone_id==g_od_vector(od).from_zone_id
            g_od_vector(od).od_gamma_numerator_error_signal...
                =g_ozone_vector(o).estimated_generation...
                *(1/(g_ozone_vector(o).total_gamma)); 
    %原来的代码为*(-1/(g_ozone_vector(o).total_gamma)); 
        end
    end
end
fprintf('Calculate the deriviative of the numerator of normalized gamma \n')    
 
for od=1:nb_od
    for o=1:nb_ozone
        if g_ozone_vector(o).ozone_id==g_od_vector(od).from_zone_id
            g_od_vector(od).od_gamma_dominator_error_signal...
                =g_ozone_vector(o).estimated_generation...
                * g_od_vector(od).od_estimated_gamma...% 这里是已经标准化过的Gamma
                *(-1/(g_ozone_vector(o).total_gamma));
  %原来的代码为 *(-1/(g_ozone_vector(o).total_gamma)^2);
        end
    end
end   
fprintf('Calculate the deriviative of the dominator of normalized gamma \n')


end