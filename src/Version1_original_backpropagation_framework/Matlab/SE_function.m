function [total_error,survey_error,cell_error,sensor_error,total_mape]=SE_function(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector,sample_ozone,sample_od,sample_link,Lambda_household, Lambda_mobile,Lambda_sensor)
[~,nb_ozone]=size(g_ozone_vector);
[~,nb_od]=size(g_od_vector);
[~,nb_path]=size(g_path_vector);
[~,nb_link]=size(g_link_vector);
[~,nb_ozone_sample]=size(sample_ozone);
[~,nb_od_sample]=size(sample_od);
[~,nb_link_sample]=size(sample_link);

% square error for residential survey
survey_error=0;
survey_mape=0;
for s=1:nb_ozone_sample
    for o=1:nb_ozone
        survey_error=survey_error+(1/2)*(g_ozone_vector(o).estimated_generation/sample_ozone{s}(o).target_generation-1)^2;
        survey_mape=survey_mape+abs((sample_ozone{s}(o).target_generation-g_ozone_vector(o).estimated_generation)/sample_ozone{s}(o).target_generation);
    end
end
survey_mape=survey_mape/nb_ozone;
% square error for cell phone data
cell_error=0;
cell_mape=0;
for s=1:nb_od_sample
for od=1:nb_od
    for o=1:nb_ozone
        if g_od_vector(od).from_zone_id==g_ozone_vector(o).ozone_id
            cell_error=cell_error+(1/2)*(g_od_vector(od).od_estimated_gamma/sample_od{s}(od).od_target_gamma-1)^2;   
            cell_mape=cell_mape+abs((sample_od{s}(od).od_target_gamma-g_od_vector(od).od_estimated_gamma)/sample_od{s}(od).od_target_gamma);
        end
    end
end
end

cell_mape=cell_mape/nb_od;
% square error for sensor data
sensor_error=0;
sensor_mape=0;
nb_sensor=0;
for s=1:nb_link_sample
for li=1:nb_link
    if size(str2num(g_link_vector(li).count_sensor_name{1}),1)~=1
        sensor_error=sensor_error+(1/2)*(g_link_vector(li).estimated_flow/sample_link{s}(li).target_flow-1)^2;
        sensor_mape=sensor_mape+abs((sample_link{s}(li).target_flow-g_link_vector(li).estimated_flow)/sample_link{s}(li).target_flow);
        nb_sensor=nb_sensor+1;
    end
end
end
sensor_mape=sensor_mape/nb_sensor;

survey_error= (Lambda_household*survey_error/nb_ozone_sample);
cell_error =(Lambda_mobile*cell_error/nb_od_sample);
sensor_error=(Lambda_sensor*sensor_error/nb_link_sample);
total_error=survey_error+cell_error+sensor_error;
total_mape=(survey_mape+cell_mape+sensor_mape)/3;
% fprintf('Calculate the square error function, total_error = %d\n', total_error)
% fprintf('Calculate the mean absolute percentage error , total_mape = %d\n', total_mape)
end