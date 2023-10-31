#ifndef TAB1DB_H
#define TAB1DB_H

#include <QWidget>
#include <QCheckBox>
#include <QTimer>
#include <QDebug>
#include <QDial>
#include <QListWidget>
#include <QListWidgetItem>
#include <QList>
#include <QListData>
#include <QStringListModel>
#include <QTextBlock>
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
    QCheckBox* pCBkeys[10];

private slots:
    void on_pPBClear_clicked();
    void on_pPBStart_clicked();
};

#endif // TAB1DB_H
