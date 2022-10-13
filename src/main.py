from database import Database
import json


def main():
    f_config = open('./config.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()

    db = Database(config_dic)
    db.scan_files()


if __name__ == '__main__':
    main()
