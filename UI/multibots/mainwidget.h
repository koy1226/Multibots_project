#ifndef __QTMAIN_H
#define __QTMAIN_H

#include <QWidget>
#include "tab1db.h"
#include "tab2cam.h"
#include "tab3socketclient.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWidget; }
QT_END_NAMESPACE

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    MainWidget(int argc, char **argv, QWidget *parent = nullptr);
    ~MainWidget();
private:
    Ui::MainWidget* ui;
    Tab1DB* pTab1DB;
    Tab2Cam* pTab2Cam;
    Tab3SocketClient* pTab3SocketClient;

};
#endif // MAINWIDGET_H
