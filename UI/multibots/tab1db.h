#ifndef TAB1DB_H
#define TAB1DB_H

#include <QWidget>
#include <QTimer>
#include <QDebug>
#include <QDial>
#include <QListWidget>
#include <QListWidgetItem>
#include <QList>
#include <QListData>
#include <QStringListModel>
#include <QTextBlock>
#include <QtSql>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QDateTime>
#include <QDebug>

namespace Ui {
class Tab1DB;
}

class Tab1DB : public QWidget
{
    Q_OBJECT

public:
    explicit Tab1DB(QWidget *parent = nullptr);
    ~Tab1DB();
    QDial * getpQDial();

private:
    Ui::Tab1DB *ui;
    QTimer * pQTimer;
    QListWidgetItem *newItem;
    QListWidget *listWidget;
    QListView listView;
    QStringList shoppinglist;
    QListData *listdata;
    QSqlDatabase db;


private slots:
    void connectToDatabase();
    void on_pCBkey8_stateChanged(int arg1);
    void on_pCBkey7_stateChanged(int arg1);
    void on_pCBkey1_stateChanged(int arg1);
    void on_pCBkey2_stateChanged(int arg1);
    void on_pCBkey3_stateChanged(int arg1);
    void on_pCBkey4_stateChanged(int arg1);
    void on_pCBkey5_stateChanged(int arg1);
    void on_pCBkey6_stateChanged(int arg1);
    void on_pCBkey9_stateChanged(int arg1);
    void on_pCBkey10_stateChanged(int arg1);
    void on_pCBkey11_stateChanged(int arg1);
    void on_pCBkey12_stateChanged(int arg1);
    void on_pPBClear_clicked();
    void on_pCBkey_stateChanged(int index, int arg1);
    void on_pPBStart_clicked();
};

#endif // TAB1DB_H
