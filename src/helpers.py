# No functions

##############################################################################################

month_num2name = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September',10: 'October', 11: 'November', 12: 'December'}
month_name2num = {n:i+1 for i,n in enumerate(list(month_num2name.values()))}

##############################################################################################

sensors_units = {'HYDRO_M2-OL-HWL.MC@MF': 'ft',
 'HYDRO_M2-GOV-SI-001-SI.MC@MF': 'RPM',
 'HYDRO_M2-GOV-SI-002-SI.MC@MF': 'RPM',
 'HYDRO_M2-DPM-GEN-MW.MC@MF': 'MW',
 'HYDRO_M2-GEN-LBRG-TE-001-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-LBRG-TE-002-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-TBRG-TE-001-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-TBRG-TE-002-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-TBRG-TE-003-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-TBRG-TE-004-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-TBRG-TE-101-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-TBRG-TE-102-F.MC@MF': 'Deg F',
 'HYDRO_M2-TURB-HYD-TE-001-F.MC@MF': 'Deg F',
 'HYDRO_M2-TURB-HYD-TE-002-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-STAT-TE-001.MC@MF': 'Deg C',
 'HYDRO_M2-GEN-STAT-TE-003.MC@MF': 'Deg C',
 'HYDRO_M2-GEN-STAT-TE-006.MC@MF': 'Deg C',
 'HYDRO_M2-GEN-STAT-TE-008.MC@MF': 'Deg C',
 'HYDRO_M2-GEN-STAT-TE-009.MC@MF': 'Deg C',
 'HYDRO_M2-GEN-STAT-TE-011.MC@MF': 'Deg C',
 'HYDRO_M2-TE-900-CUR.MC@MF': 'Deg C',
 'HYDRO_M2-GEN-HTX-TE-001-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-HTX-TE-002-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-HTX-TE-003-F.MC@MF': 'Deg F',
 'HYDRO_M2-GEN-HTX-TE-004-F.MC@MF': 'Deg F',
 'HYDRO_M2-EXC-001-TI-FIELD-F.MC@MF': 'Deg F',
 'HYDRO_M2_BN_GGB_X_DA.MC@MF': 'mils',
 'HYDRO_M2_BN_GGB_Y_DA.MC@MF': 'mils',
 'HYDRO_M2_BN_TGB_X_DA.MC@MF': 'mils',
 'HYDRO_M2_BN_TGB_Y_DA.MC@MF': 'mils',
 'HYDRO_M2-DPM-GEN-MVAR.MC@MF': 'mvar',
 'HYDRO_M2-SI-FXD3.MC@MF': 'gen_frequency',
 'HYDRO_M2-REL-25G-XI-ANGLE.MC@MF': 'DEGREE',
 'HYDRO_M2-EXC-001-II-FIELD.MC@MF': 'Amps',
 'HYDRO_M2-EXC-001-EI-FIELD.MC@MF': 'VDC',
 'HYDRO_M2-INLETDIS-HOV-001.MC@MF': 'ft',
 'HYDRO_M2-GOV-WG-OPOS.MC@MF': 'PERCENTAGE (%)',
 'HYDRO_M2-GOV-GO-PT-002.MC@MF': 'PSIG',
 'HYDRO_M2-INLETDIS-PT-001.MC@MF': 'pressure_scroll_case',
 'HYDRO_M2-BA-PT-001.MC@MF': 'pressure_air_brake',
 'HYDRO_R1-DA-PI-003.R1@RV': 'pressure_air_depressing',
 'HYDRO_M2-DA-PT-001.MC@MF': 'pressure_air_depression',
 'index': 'Date',
 'Unnamed: 0': 'Date'}

##############################################################################################

# usecase_InstrumentID dictionary
usecase_iid = {'water_level': 'HYDRO_M2-OL-HWL.MC@MF',
               'rpm1': 'HYDRO_M2-GOV-SI-001-SI.MC@MF',
               'rpm2': 'HYDRO_M2-GOV-SI-002-SI.MC@MF',
               'pwr': 'HYDRO_M2-DPM-GEN-MW.MC@MF',
               'temp_gen_guide_bearing001': 'HYDRO_M2-GEN-LBRG-TE-001-F.MC@MF',
               'temp_gen_guide_bearing002': 'HYDRO_M2-GEN-LBRG-TE-002-F.MC@MF',
               'temp_thrust_bearing001': 'HYDRO_M2-GEN-TBRG-TE-001-F.MC@MF',
               'temp_thrust_bearing002': 'HYDRO_M2-GEN-TBRG-TE-002-F.MC@MF',
               'temp_thrust_bearing003': 'HYDRO_M2-GEN-TBRG-TE-003-F.MC@MF',
               'temp_thrust_bearing004': 'HYDRO_M2-GEN-TBRG-TE-004-F.MC@MF',
               'temp_thrust_bearing_oil101': 'HYDRO_M2-GEN-TBRG-TE-101-F.MC@MF',
               'temp_thrust_bearing_oil102': 'HYDRO_M2-GEN-TBRG-TE-102-F.MC@MF',
               'temp_turb_guide_bearing001': 'HYDRO_M2-TURB-HYD-TE-001-F.MC@MF',
               'temp_turb_guide_bearing002': 'HYDRO_M2-TURB-HYD-TE-002-F.MC@MF',
               'temp_stator001': 'HYDRO_M2-GEN-STAT-TE-001.MC@MF',
               'temp_stator003': 'HYDRO_M2-GEN-STAT-TE-003.MC@MF',
               'temp_stator006': 'HYDRO_M2-GEN-STAT-TE-006.MC@MF',
               'temp_stator008': 'HYDRO_M2-GEN-STAT-TE-008.MC@MF',
               'temp_stator009': 'HYDRO_M2-GEN-STAT-TE-009.MC@MF',
               'temp_stator011': 'HYDRO_M2-GEN-STAT-TE-011.MC@MF',
               'temp_ambient': 'HYDRO_M2-TE-900-CUR.MC@MF',
               'temp_air_cooler001': 'HYDRO_M2-GEN-HTX-TE-001-F.MC@MF',
               'temp_air_cooler002': 'HYDRO_M2-GEN-HTX-TE-002-F.MC@MF',
               'temp_air_cooler003': 'HYDRO_M2-GEN-HTX-TE-003-F.MC@MF',
               'temp_air_cooler004': 'HYDRO_M2-GEN-HTX-TE-004-F.MC@MF',
               'temp_exciter': 'HYDRO_M2-EXC-001-TI-FIELD-F.MC@MF',
               'vib_upper_guide_bearingX': 'HYDRO_M2_BN_GGB_X_DA.MC@MF',
               'vib_upper_guide_bearingY': 'HYDRO_M2_BN_GGB_Y_DA.MC@MF',
               'vib_turbine_guide_bearingX': 'HYDRO_M2_BN_TGB_X_DA.MC@MF',
               'vib_turbine_guide_bearingY': 'HYDRO_M2_BN_TGB_Y_DA.MC@MF',               
               'mvar': 'HYDRO_M2-DPM-GEN-MVAR.MC@MF',
               'gen_frequency': 'HYDRO_M2-SI-FXD3.MC@MF',
               'angle_sync': 'HYDRO_M2-REL-25G-XI-ANGLE.MC@MF',
               'amperate_exciter': 'HYDRO_M2-EXC-001-II-FIELD.MC@MF',
               'voltage_exciter': 'HYDRO_M2-EXC-001-EI-FIELD.MC@MF',
               'position_inlet_gate': 'HYDRO_M2-INLETDIS-HOV-001.MC@MF',
               'position_wicket_gate': 'HYDRO_M2-GOV-WG-OPOS.MC@MF',
               'pressure_governor_oil': 'HYDRO_M2-GOV-GO-PT-002.MC@MF',
               'pressure_scroll_case': 'HYDRO_M2-INLETDIS-PT-001.MC@MF',
               'pressure_air_brake': 'HYDRO_M2-BA-PT-001.MC@MF',
               'pressure_air_depressing': 'HYDRO_R1-DA-PI-003.R1@RV',
               'pressure_air_depression': 'HYDRO_M2-DA-PT-001.MC@MF',
              } 
# Reverse mapping
iid_usecase = {v:k for k,v in usecase_iid.items()}

##############################################################################################
normal_timeframes = [['2020-11-09', '2020-11-25'],['2020-12-04', '2021-01-03'],
          ['2021-01-04', '2021-03-28'],['2021-04-03', '2022-01-17'],
          ['2022-02-11', '2022-04-07'], ['2022-04-08', '2022-11-28'],
          ['2022-12-21', '2023-02-26'],['2023-03-10', '2023-09-05']]


##############################################################################################