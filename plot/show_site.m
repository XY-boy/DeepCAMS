clc;close all; clear;
site_dir1 = 'D:\Github-package\SR_PM2.5\Dataset\站点数据\2019\';
site_dir2 = 'D:\Github-package\SR_PM2.5\Dataset\站点数据\2018\';
site_dir3 = 'D:\Github-package\SR_PM2.5\Dataset\站点数据\2017\';

txt_list1 = dir([site_dir1, '*.txt']);
txt_list2 = dir([site_dir2, '*.txt']);
txt_list3 = dir([site_dir3, '*.txt']);


figure;
set(gcf,'outerposition',get(0,'screensize'));
earthimage('watercolor',[176,224,230]/255.);
borders('countries','color',0.8*[1 1 1]);

xticklabels({'150° W','100° W','50° W','0° ','50° E','100° E','150° E'});
yticklabels({'80° S','60° S','40° S','20° S','0° ','20° N','40° N','60° N','80° N'});

set(gca,'FontSize',20,'Fontname','times new Roman')

for i=1:1:length(txt_list1)
    [txt_list1(i).folder, '\',txt_list1(i).name]
    txt_data = load([txt_list1(i).folder, '\',txt_list1(i).name]);
    lat = txt_data(:,5);
    lon = txt_data(:,6);
    scatter(lon,lat,8,'filled','MarkerFaceColor',[255,0,255]/255);
    hold on
end

for i=1:1:length(txt_list2)
    [txt_list2(i).folder, '\',txt_list2(i).name]
    txt_data2 = load([txt_list2(i).folder, '\',txt_list2(i).name]);
    lat2 = txt_data2(:,5);
    lon2 = txt_data2(:,6);
    scatter(lon2,lat2,8,'filled','MarkerFaceColor',[255,0,255]/255);
    hold on
end

for i=1:1:length(txt_list3)
    [txt_list3(i).folder, '\',txt_list3(i).name]
    txt_data3 = load([txt_list3(i).folder, '\',txt_list3(i).name]);
    lat3 = txt_data3(:,5);
    lon3 = txt_data3(:,6);
    scatter(lon3,lat3,8,'filled','MarkerFaceColor',[255,0,255]/255);
    hold on
end
% exportgraphics(gcf,'site.png','Resolution',300);
