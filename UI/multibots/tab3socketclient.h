#ifndef TAB3SOCKETCLIENT_H
#define TAB3SOCKETCLIENT_H

#include <QWidget>
#include <QTime>
#include "socketclient.h"
#include <string>

namespace Ui {
class Tab3SocketClient;
}

class Tab3SocketClient : public QWidget
{
    Q_OBJECT

public:
    explicit Tab3SocketClient(QWidget *parent = nullptr);
    ~Tab3SocketClient();

private:
    Ui::Tab3SocketClient *ui;
    SocketClient * pSocketClient;

private slots:
    void slotConnectToServer(bool);
    void slotSocketRecvUpdate(QString);
    void slotSocketSendData();
    void slotSocketSendData(QString);
signals:
    void sigLedWrite(int);
    void goGoalSig(double, double, double, double);

};

#endif // Tab3SocketClient_H
