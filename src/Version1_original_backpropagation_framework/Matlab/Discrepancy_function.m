function [total_error,survey_error,ozone_ef,ozone_tf,cell_error,od_ef,od_tf,sensor_error,link_ef,link_tf]=Discrepancy_function(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector)
[~,nb_ozone]=size(g_ozone_vector);
[~,nb_od]=size(g_od_vector);
[~,nb_path]=size(g_path_vector);
[~,nb_link]=size(g_link_vector);

% square error for residential survey
survey_error=0;
ozone_tf=[];
ozone_ef=[];
for o=1:nb_ozone
    survey_error=survey_error+(1/2)*(g_ozone_vector(o).estimated_generation-g_ozone_vector(o).target_generation)^2;
    ozone_tf=[ozone_tf,g_ozone_vector(o).target_generation];
    ozone_ef=[ozone_ef,g_ozone_vector(o).estimated_generation];
end
1
% square error for cell phone data
od_tf=[];
od_ef=[];
cell_error=0;
for od=1:nb_od
    for o=1:nb_ozone
        if g_od_vector(od).from_zone_id==g_ozone_vector(o).ozone_id
            cell_error=cell_error+(1/2)*(g_od_vector(od).od_estimated_gamma-g_od_vector(od).od_target_gamma)^2;
            od_ef=[od_ef,g_od_vector(od).estimated_demand];
            od_tf=[od_tf,g_od_vector(od).target_demand];
        end
    end
end
% square error for sensor data
sensor_error=0;
nb_sensor=0;
link_ef=[];
link_tf=[];
for li=1:nb_link
    if size(str2num(g_link_vector(li).count_sensor_name{1}),1)~=1
        sensor_error=sensor_error+(1/2)*(g_link_vector(li).estimated_flow-g_link_vector(li).target_flow)^2;
        link_ef=[link_ef,g_link_vector(li).estimated_flow];
        link_tf=[link_tf,g_link_vector(li).target_flow];
        nb_sensor=nb_sensor+1;
    end
end
total_error=sensor_error+cell_error+survey_error;
fprintf('Calculate the discrepancy function, total_error = %d\n', total_error)
end