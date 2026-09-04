#ifndef GATE_COMM_H
#define GATE_COMM_H

#include <stdint.h>
#pragma once
#include <sqlite3.h>
#include <string>

int GetMaxValidCount();
bool GetHasUmbrella();
void UpdateUmbrellaStatus(bool has_umbrella);
sqlite3* GetDatabaseConnection(const std::string& db_name);
void init_serial_comm(const char* device);
void send_people_count(uint16_t people, bool has_umbrella);
void send_people_count(uint16_t people);
void update_max_info(const std::string& device_id, const std::string& timestamp);

#endif // GATE_COMM_H
